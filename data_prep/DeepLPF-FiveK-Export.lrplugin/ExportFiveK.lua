--[[
Export the FiveK Input and Expert-C collections for DeepLPF.

Presents a dialog listing every collection in the open catalogue, lets you pick
which is the Input and which is the Expert-C target, then exports each to
~/fivek_export/input and ~/fivek_export/output as PNG / sRGB / 8-bit /
long-edge-512 (no enlarge), preserving the original "aXXXX-..." filenames.
]]

local LrApplication      = import 'LrApplication'
local LrTasks            = import 'LrTasks'
local LrFunctionContext  = import 'LrFunctionContext'
local LrDialogs          = import 'LrDialogs'
local LrView             = import 'LrView'
local LrBinding          = import 'LrBinding'
local LrExportSession    = import 'LrExportSession'
local LrPathUtils        = import 'LrPathUtils'
local LrFileUtils        = import 'LrFileUtils'

local bind = LrView.bind

-- Destination folders (~/fivek_export/{input,output}); matches the repo scripts.
local HOME       = LrPathUtils.getStandardFilePath('home')
local EXPORT_DIR = LrPathUtils.child(HOME, 'fivek_export')
local INPUT_DIR  = LrPathUtils.child(EXPORT_DIR, 'input')
local OUTPUT_DIR = LrPathUtils.child(EXPORT_DIR, 'output')

-- Recursively gather every collection, labelled with its "Set / Name" path.
local function gatherCollections(parent, prefix, out)
	for _, coll in ipairs(parent:getChildCollections()) do
		out[#out + 1] = {
			name = prefix .. coll:getName(),
			id   = coll.localIdentifier,
			coll = coll,
		}
	end
	for _, set in ipairs(parent:getChildCollectionSets()) do
		gatherCollections(set, prefix .. set:getName() .. ' / ', out)
	end
end

-- The DeepLPF export preset: PNG, sRGB, 8-bit, fit within 512x512 (=> long edge
-- 512), don't enlarge, keep original filenames, overwrite on re-run.
local function exportSettings(destFolder)
	return {
		-- Plain "Export to Hard Drive"; some SDK versions require this to be named
		-- explicitly rather than defaulting.
		LR_exportServiceProvider      = 'com.adobe.ansel.hard_drive',
		LR_exportServiceProviderTitle = 'Hard Drive',

		LR_export_destinationType       = 'specificFolder',
		LR_export_destinationPathPrefix = destFolder,
		LR_export_useSubfolder          = false,
		LR_collisionHandling            = 'overwrite',

		LR_format                       = 'PNG',
		LR_export_colorSpace            = 'sRGB',
		LR_export_bitDepth              = 8,

		LR_size_doConstrain             = true,
		LR_size_doNotEnlarge            = true,
		LR_size_resizeType              = 'wh',   -- fit within maxWidth x maxHeight
		LR_size_maxWidth                = 512,
		LR_size_maxHeight               = 512,
		LR_size_units                   = 'pixels',
		LR_size_resolution              = 240,
		LR_size_resolutionUnits         = 'inch',

		LR_outputSharpeningOn           = false,
		LR_reimportExportedPhoto        = false,
		LR_renamingTokensOn             = false,  -- keep original filename
		LR_includeVideoFiles            = false,
		LR_removeLocationMetadata       = false,
		LR_embeddedMetadataOption       = 'all',
	}
end

-- Export one collection to destFolder; returns the number of photos exported.
local function exportCollection(coll, destFolder)
	LrFileUtils.createAllDirectories(destFolder)
	local photos = coll:getPhotos()
	if #photos == 0 then
		return 0
	end
	local session = LrExportSession {
		photosToExport = photos,
		exportSettings = exportSettings(destFolder),
	}
	-- Runs synchronously on this task and shows Lightroom's own progress bar.
	session:doExportOnCurrentTask()
	return #photos
end

LrFunctionContext.postAsyncTaskWithContext('DeepLPF FiveK Export', function(context)
	local catalog = LrApplication.activeCatalog()

	local collections = {}
	gatherCollections(catalog, '', collections)
	if #collections == 0 then
		LrDialogs.message('No collections found',
			'Open the FiveK catalogue (fivek.lrcat) first, then run this again.', 'warning')
		return
	end

	-- Build popup items and a lookup from localIdentifier -> collection object.
	local items, byId = {}, {}
	local defaultInput, defaultTarget
	for _, e in ipairs(collections) do
		local n = #e.coll:getPhotos()
		items[#items + 1] = { title = e.name .. '  (' .. n .. ' photos)', value = e.id }
		byId[e.id] = e.coll
		local lname = string.lower(e.name)
		-- Best-guess defaults; user can override in the dialog.
		-- InputAsShotZeroed is the rendering that matches the reference inputs,
		-- so prefer it over any other collection whose name contains 'input'.
		if lname:find('inputasshotzeroed') then defaultInput = e.id
		elseif not defaultInput and lname:find('input') then defaultInput = e.id end
		if not defaultTarget and (lname:find('/ c') or lname:find('expert c')
			or lname:match('/%s*c$')) then defaultTarget = e.id end
	end

	local props = LrBinding.makePropertyTable(context)
	props.inputColl  = defaultInput  or collections[1].id
	props.targetColl = defaultTarget or collections[1].id

	local f = LrView.osFactory()
	local contents = f:column {
		bind_to_object = props,
		spacing = f:control_spacing(),
		f:static_text { title = 'Input collection  ->  ~/fivek_export/input', font = '<system/bold>' },
		f:popup_menu { items = items, value = bind('inputColl'), width = 560 },
		f:spacer { height = 8 },
		f:static_text { title = 'Expert-C target collection  ->  ~/fivek_export/output', font = '<system/bold>' },
		f:popup_menu { items = items, value = bind('targetColl'), width = 560 },
		f:spacer { height = 8 },
		f:static_text {
			title = 'Exports as PNG / sRGB / 8-bit / long edge 512 (no enlarge),\n'
				.. 'keeping original filenames. Existing files are overwritten.',
			height_in_lines = 2,
		},
	}

	local choice = LrDialogs.presentModalDialog {
		title    = 'Export FiveK for DeepLPF',
		contents = contents,
		actionVerb = 'Export',
	}
	if choice ~= 'ok' then
		return
	end

	if props.inputColl == props.targetColl then
		LrDialogs.message('Pick two different collections',
			'The Input and Expert-C target must be different collections.', 'warning')
		return
	end

	local ok, err = LrTasks.pcall(function()
		local nIn  = exportCollection(byId[props.inputColl], INPUT_DIR)
		local nOut = exportCollection(byId[props.targetColl], OUTPUT_DIR)
		LrDialogs.message('Export complete',
			nIn .. ' inputs -> ' .. INPUT_DIR .. '\n' ..
			nOut .. ' targets -> ' .. OUTPUT_DIR .. '\n\n' ..
			'Now tell your terminal session "exported" to run organise + verify.', 'info')
	end)
	if not ok then
		LrDialogs.message('Export failed', tostring(err), 'critical')
	end
end)
