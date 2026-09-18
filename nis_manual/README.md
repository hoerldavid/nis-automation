# NIS Elements Help Manuals

NIS Elements provides an extensive manual through it's help menu. It is a standard Windows help viewer and built from Windows Compiled HTML Help files (`.chm`).
The source `.chm` files can be found in a NIS Elements installation under `docs/` e.g. `C:\Program Files\NIS-Elements\docs\`.

Among others, particularly interesting ones include:

* `NIS_AR_ENG.CHM` – Main English manual
* `nis_ar_fr_eng.chm` – Function Reference for the macro language

The `.chm` files are copyrighted by Nikon/Laboratory Imaging and are **not** included in this repository. If you have a legitimate NIS Elements installation, you can obtain the `.chm` files from the installation directory. This note describes how to extract the `.chm` files for viewing in a standard web browser.

## Extracting to HTML

A `.chm` file is essentially a compressed archive containing HTML pages, images, and index files. It can be extracted with 7-Zip / p7zip.

**Note:** You can extract multiple help `.chm` files to the same folder as the individual help pages have distinct names. A few shared assets (navigation icons, etc.) can be safely overwritten with the `-y` flag. This gives you a "combined manual" like in the NIS help menu.

### Linux / macOS

To extract, you need p7zip, which you can install e.g. via apt or Homebrew:

```bash
apt install p7zip-full
```
or

```bash
brew install p7zip
```

Then, you can extract the manual via:

```bash
mkdir nis_help_html
7z x /path/to/NIS_AR_ENG.CHM -onis_help_html
# or for the Function Reference (overwrite shared files with -y):
# 7z x /path/to/nis_ar_fr_eng.chm -onis_help_html -y
```

### Windows

With the 7-Zip desktop app installed:

* GUI: Right-click the `.chm` → 7-Zip → Extract to "NIS_AR_ENG.CHM\" or Extract to... and choose a folder, e.g. a local `nis_help_html` directory.
* Command line: 7-Zip installs `7z.exe`, typically `C:\Program Files\7-Zip\7z.exe`.

```powershell
mkdir nis_help_html
"C:\Program Files\7-Zip\7z.exe" x C:\Program Files\Nikon\NIS-Elements\docs\NIS_AR_ENG.CHM -onis_help_html -y
"C:\Program Files\7-Zip\7z.exe" x C:\Program Files\Nikon\NIS-Elements\docs\nis_ar_fr_eng.chm -onis_help_html -y
```

### Using extracted files

If you unpack both manuals as described above, you will have ~8.7k unique files total, including:

* `NIS_AR_ENG.CHM` → ~5k files
* `nis_ar_fr_eng.chm` → ~3.7k files, with ~40 overlapping metadata/shared files

The extracted tree contains `*.html` files for individual help topics. Navigation partially depends on `.chm` index files, so it's best to search via `grep`, etc. and open individual files.

