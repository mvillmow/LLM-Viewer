from flask import Flask, request
from flask import render_template
from flask_cors import CORS
from get_model_graph import get_model_graph
from backend_settings import avaliable_hardwares, suggested_model_ids
import argparse

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def index():
    return "backend server ready."


@app.route("/get_graph", methods=["POST"])
def get_graph():
    inference_config = request.json["inference_config"]
    model_id = request.json["model_id"]
    hardware = request.json["hardware"]
    config_path = request.json.get("config_path", None)  # Optional custom config
    source = request.json.get("source", None)  # Optional custom source
    
    nodes, edges, total_results, hardware_info = get_model_graph(
        model_id,
        hardware,
        config_path,
        inference_config,
        source,
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "total_results": total_results,
        "hardware_info": hardware_info,
    }

@app.route("/get_avaliable", methods=["GET"])
def get_avaliable():
    return {
        "avaliable_hardwares": avaliable_hardwares,
        "suggested_model_ids": suggested_model_ids,
    }

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args=parser.parse_args()
    host="127.0.0.1" if args.local else "0.0.0.0"
    app.run(debug=args.debug,host=host,port=args.port)
