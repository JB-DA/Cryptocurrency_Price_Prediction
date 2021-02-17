
from flask import Flask, jsonify, render_template, url_for, redirect 

##### DATABASE SETUP
###
#
#engine = create_engine(f"postgresql://postgres:postgres@localhost:5432/crypto_analysis_db")
#Base = automap_base()
#Base.prepare(engine, reflect=True)


##### FLASK SETUP
###
#
app = Flask(__name__, template_folder='static', static_folder='static')


##### ROUTES
###
#
# API Pages
# @app.route("/")
# def api_overview():
#     return jsonify(json_overview)


# HTML PAGES
@app.route('/')
def index():
    return render_template('index.html')

##### RUN APP
###
#
if __name__ == '__main__':
    app.run(debug=True)