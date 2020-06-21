from flask import Flask, request
from flask_restful import Resource, Api
import requests

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}
class TransforMap(Resource):
    def post(self):
        eta = requests('<MODEL_PREDICTION_URL>', request.json)
        return eta.json()

api.add_resource(HelloWorld, '/')
api.add_resource(TransforMap, '/prediction')

if __name__ == '__main__':
    app.run(debug=True)