from flask import Flask, render_template
import nbconvert
import nbformat
import os

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/project_PINNs')
def project_PINNs():
    notebook_html = convert_notebook_to_html('PINNs_notebook.ipynb')
    return render_template('projects_PINNs.html',notebook_content=notebook_html)

if __name__ == '__main__':
    app.run(debug=True)