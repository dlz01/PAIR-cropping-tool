# PAIR-cropping-tool

This interactive tool allows users to select a quadrilateral region within an image, crop this area, and save the cropped image. It's available for Windows and macOS, as well as in source code form for those who wish to run it directly with Python.

## Download

You can download the tool directly from our website:

- For **Windows**: Download the `.exe` installer.
- For **macOS**: Download the `.dmg` file.
- **Source Code**: If you prefer to run the tool directly from the source code using Python, download the `.py` file.

Download link: 
[Windows](https://raw.githubusercontent.com/dlz01/PAIR-cropping-tool/main/mark_and_crop.exe), 
[macOS](https://raw.githubusercontent.com/dlz01/PAIR-cropping-tool/main/mark_and_crop.dmg),
[Source](https://raw.githubusercontent.com/dlz01/PAIR-cropping-tool/main/mark_and_crop.py)

## Installation

### Windows

1. Download the `.exe` file from the website.
2. Double-click the downloaded file to start the installation process.
3. Follow the on-screen instructions to install the tool.

### macOS

1. Download the `.dmg` file from the website.
2. Click the `.dmg` file to open a pop-up window
3. Double-click the application mark_and_crop in the pop-up window to run it.
4. If you encounter a warning about an "unverified developer," follow the steps below to open the app:

    a. Click cancel when the warning pop up.
    
    b. Open "System Preferences" and navigate to "Security & Privacy".
    
    c. Under "General" tab, there is an option to open the app anyway. Click "Open."

    This process only needs to be done once. Afterward, you can open the app as usual.

### From Source Code

Ensure you have Python installed on your system along with the necessary packages:

```bash
pip install numpy opencv-python
```

Then, run the script from the terminal or command prompt:

```bash
python mark_and_crop.py
```

## Instruction

After installation, you can start the tool from your applications list on Windows or macOS. If you're running from the source code, follow the instructions in the From Source Code section.

1. **Select folder**: Select the folder containing the images that you want to crop.
1. **Select Points**: Click on the displayed image to select the four corners of the quadrilateral region you wish to crop.
2. **Adjust Selection**: Right-click near a point to remove the closest selected point. Right-double-click to remove all selected points. Continue selecting until you have defined all four corners.
3. **Use Auto Marker**: Press the 'm' key to enable or disable the auto point marker.
4. **Crop and Save**: Press the 's' key to crop and save the selected region.
5. **Skip**: Press the 'esc' key to skip one image, it will show up again the next time you open this tool. 
6. **Exit**: Press the 'p' key to pause and exit. The tool will not revisit cropped images, if you want to edit some images again, you can manually delete the corresponding cropped images in the working folder. 