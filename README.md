# PAIR-cropping-tool

This interactive tool allows users to select a quadrilateral region within an image, crop this area, and save the cropped image. It's available for Windows and macOS, as well as in source code form for those who wish to run it directly with Python.

## Download

You can download the tool directly from our website:

- For **Windows**: Download the `.exe` installer.
- For **macOS**: Download the `.dmg` file.
- **Source Code**: If you prefer to run the tool directly from the source code using Python, download the `.py` file.

[Download Link](#)

## Installation

### Windows

1. Download the `.exe` file from the website.
2. Double-click the downloaded file to start the installation process.
3. Follow the on-screen instructions to install the tool.

### macOS

1. Download the `.dmg` file from the website.
2. Double-click the `.dmg` file to open a pop-up window
3. Double-click the application in the pop-up window to run it.

### From Source Code

Ensure you have Python installed on your system along with the necessary packages:

```bash
pip install numpy opencv-python
```

Then, run the script from the terminal or command prompt:

```bash
python mark_and_crop.py
```

## Usage

After installation, you can start the tool from your applications list on Windows or macOS. If you're running from the source code, follow the instructions in the From Source Code section.

1. **Select Points**: Click on the displayed image to select the four corners of the quadrilateral region you wish to crop.
2. **Adjust Selection**: Right-click near a point to remove the closest selected point. Right-double-click to remove all selected points. Continue selecting until you have defined all four corners.
3. **Use Auto Marker**: Press the 'm' key to enable or disable the auto point marker.
4. **Crop and Save**: Press the 's' key to crop and save the selected region.
5. **Exit**: Press the 'p' key to pause and exit. 