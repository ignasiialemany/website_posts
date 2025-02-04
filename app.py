from flask import Flask, render_template
import nbconvert
import nbformat
import os
import json
from bs4 import BeautifulSoup

app = Flask(__name__)

def convert_notebook_to_html(ipynb_path):
    try:
        with open(ipynb_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        html_exporter = nbconvert.HTMLExporter()
        html_exporter.template_name = 'classic'
        
        body, _ = html_exporter.from_notebook_node(nb)
        return body
    except Exception as e:
        return f"Error loading notebook: {str(e)}"


def load_project(project_name):
    with open(project_name, 'r') as file:
        return json.load(file)["projects"]

@app.route('/')
def index():
    projects = load_project("projects.json")
    return render_template('index.html',projects=projects)

@app.route('/projects_<int:project_index>')
def project_page(project_index):
    projects = load_project("projects.json")
    print(projects)
    project = projects[project_index]
    with open(project["notebook_file"], 'r') as file:
        notebook_html = file.read()
    return render_template('projects.html',title=project["title"],description=project["description"],notebook_content=notebook_html)

if __name__ == '__main__':
    app.run(debug=True)