from flask import Flask, render_template
import nbconvert
import nbformat
import os
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bayesian_linear_regressions')
def project_bayesian_linear_regressions():
    with open('BayesianLinearRegression.html', 'r') as file:
        notebook_html = file.read()
    soup = BeautifulSoup(notebook_html, 'html.parser')
    body_content = soup.body.encode_contents().decode()
    return render_template('BayesianLinearRegression.html',notebook_content=notebook_html,title="Bayesian Linear Regressions")

@app.route('/project_PINNs')
def project_PINNs():
    return render_template('PCA.html')

if __name__ == '__main__':
    app.run(debug=True)