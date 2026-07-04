# fractal-explorer
Fun 3d fractal exploration

## How to Run Locally

This is a static web project. There is no build step required, but you should serve the files using a local web server to avoid browser security restrictions (CORS) when loading local assets.

**Option 1: Python (Recommended)**
If you have Python installed, open your terminal in the project folder and run:
```bash
python3 -m http.server 8080
```
Then navigate to `http://localhost:8080` in your web browser.

**Option 2: Node.js / npm**
If you have Node installed, you can use `serve`:
```bash
npx serve .
```

**Option 3: VS Code**
If you use Visual Studio Code, you can install the **Live Server** extension, right-click `index.html`, and select "Open with Live Server".