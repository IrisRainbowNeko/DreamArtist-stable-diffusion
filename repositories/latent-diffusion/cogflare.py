from pydantic import ValidationError
from fastapi.responses import JSONResponse
import websocket
import logging
import _thread
import time
import json
import rel
import io, requests

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from cog.predictor import get_input_type, load_config, load_predictor

from cog.response import Status
from cog.server.runner import PredictionRunner
from cog.json import upload_files

logger = logging.getLogger("cog")


def run():
    runner = PredictionRunner()
    InputType = get_input_type(predictor)
    runner.setup()
    upload_url = "https://cog.nmb.ai/v1/upload"

    def upload(obj: Any) -> Any:
        def upload_file(fh: io.IOBase) -> str:
            resp = requests.put(upload_url, files={"file": fh})
            resp.raise_for_status()
            return resp.json()["url"]

        return upload_files(obj, upload_file)

    def on_message(ws, message):
        request = json.loads(message)
        try:
            input_obj = InputType(**request["input"])
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            return

        print(input_obj)
        logs: List[str] = []
        response: Dict[str, Any] = {"logs": logs}
        runner.run(**input_obj.dict())

        while runner.is_processing() and not runner.has_output_waiting():
            if runner.has_logs_waiting():
                logs.extend(runner.read_logs())
                ws.send(json.dumps(response))

        if runner.error() is not None:
            logger.error(runner.error())
            response["status"] = Status.FAILED
            response["error"] = e
            ws.send(json.dumps(response))
            return

        if runner.is_output_generator():
            output = response["output"] = []

            while runner.is_processing():
                if runner.has_output_waiting() or runner.has_logs_waiting():
                    new_output = [upload(o) for o in runner.read_output()]
                    new_logs = runner.read_logs()

                    if new_output == [] and new_logs == []:
                        continue

                    output.extend(new_output)
                    logs.extend(new_logs)
                    ws.send(json.dumps(response))
            if runner.error() is not None:
                response["status"] = Status.FAILED
                response["error"] = str(runner.error)
                ws.send(json.dumps(response))
                return

            response["status"] = Status.SUCCEEDED
            output.extend(upload(o) for o in runner.read_output())
            logs.extend(runner.read_logs())
            ws.send(json.dumps(response))
        else:
            while runner.is_processing():
                if runner.has_logs_waiting():
                    logs.extend(runner.read_logs())
                    ws.send(json.dumps(response))
            if runner.error() is not None:
                response["status"] = Status.FAILED
                response["error"] = str(runner.error())
                ws.send(json.dumps(response))
            output = runner.read_output()

            assert len(output) == 1

            response["status"] = Status.SUCCEEDED
            response["output"] = upload(output[0])
            logs.extend(runner.read_logs())
            ws.send(json.dumps(response))

    def on_error(ws, error):
        print(error)

    def on_close(ws, close_status_code, close_msg):
        print("### closed ###")

    def on_open(ws):
        print("Opened connection")

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        "wss://cog.nmb.ai/v1/queue/TEST/websocket",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever(dispatcher=rel)
    rel.signal(2, rel.abort)
    rel.dispatch()


if __name__ == "__main__":
    config = load_config()
    predictor = load_predictor(config)
    run()
