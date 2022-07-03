from flask import Flask,request,jsonify
from matplotlib.pyplot import title
from numpy import tile
import main
from main import recommendations

app = Flask(__name__)


@app.route("/movie",methods=["POST"])
def movie_rec():
    req = request.json

    try:
        movie_name = req['movie_name']
        txnid = req['txnid']

        data1 = []
        data={
            "recommendation_list":recommendations(movie_name, cosine_sim = main.similarity)
        }
        data1.append(data)
        return ({"respcode":"200","respdesc":"success","data":data1})
    except Exception as e:
        print(f'The error is:{e}')
        return ({"respcode":"404","respdesc":"failed","data":None})


if __name__ =="__main__":
    app.run(debug=True)