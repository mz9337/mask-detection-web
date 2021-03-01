# Install
### Requirements
-   Node.js
-   Python 3.6
-   pip

### Preparing enviroment

1. Clone repository
2. Run: `npm install`
3. Create pip venv: `python -m venv venv` and activate it (you can choose other name, but venv is already in .gitignore)
4. Install dependencies from requirements.txt: `pip install -r requirements.txt`
5. Beacuse of size, models were not added to repository and must be placed manually. Place `COVID19-v2_resnet152_9.pt` to `app\prediction` and `Resnet50_Final.pth` to `app\prediction\weights`. You can download models here: https://drive.google.com/file/d/1Zb99ETi49uUZJikB3X3-wLryNiFfhq1E/view?usp=sharing

### Running project
1. To run project, first build frontend assets(js + css):`npm run build` (this is only needed first time you run the project or if you change any js or css). To watch js or css for any changes, run: `npm start`
2. To run backend of the project, run: `flask run` (first time it takes some time to run). App will be available on: http://localhost:5000/ 