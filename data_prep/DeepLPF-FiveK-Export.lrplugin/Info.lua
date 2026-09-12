--[[
DeepLPF FiveK Export -- Lightroom Classic plug-in.

Adds "Export FiveK for DeepLPF..." to File > Plug-in Extras (and Library >
Plug-in Extras). It exports a chosen Input collection and a chosen Expert-C
target collection to ~/fivek_export/input and ~/fivek_export/output with the
exact settings DeepLPF expects: PNG, sRGB, 8-bit, long edge 512, no enlarge.

See README.md in this folder for install/run instructions.
]]

return {
	LrSdkVersion = 10.0,
	LrSdkMinimumVersion = 6.0,
	LrToolkitIdentifier = 'com.deeplpf.fivekexport',
	LrPluginName = 'DeepLPF FiveK Export',

	-- File > Plug-in Extras
	LrExportMenuItems = {
		{ title = 'Export FiveK for DeepLPF\226\128\166', file = 'ExportFiveK.lua' },
	},
	-- Library > Plug-in Extras (same action, reachable from the Library module)
	LrLibraryMenuItems = {
		{ title = 'Export FiveK for DeepLPF\226\128\166', file = 'ExportFiveK.lua' },
	},

	VERSION = { major = 1, minor = 0, revision = 0 },
}
