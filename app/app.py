from flask import Flask , render_template,request
from datetime import datetime
import numpy as np
from recommend import *
app = Flask(__name__)


@app.route("/",methods=['GET', 'POST'])
def home():
    titles = getTitles()
    titles = [i for i in titles]
    return render_template ('app.html',val = titles)
@app.route("/recommend",methods=['GET', 'POST'])
def recommend():
    results = request.form.get('search-term')
    # titles,imgs = authors_recommendations(results)[0], authors_recommendations(results)[1]
    # titles = [i for i in titles]
    # imgs = [i for i in imgs]
    query_book,recommendations =  authors_recommendations(results)
    return render_template ('recommend2.html',res=recommendations,query=results,query_book=query_book)
if __name__ == "__main__":
	app.run(debug=True)