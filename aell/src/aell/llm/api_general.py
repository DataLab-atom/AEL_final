import ctypes
import http.client
import json
import time
from threading import Thread

import openai

def get_response(api_endpoint, payload_explanation, headers, out_list):
    try:
        conn = http.client.HTTPSConnection(api_endpoint)
        conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
        res = conn.getresponse()
        data = res.read()
        json_data = json.loads(data)
        response = json_data["choices"][0]["message"]["content"]
        out_list[0] = response  # 数据传回
    except Exception as e:
        print(f"Error in API. {e} ")

# 用于强制终止线程的函数
def terminate_thread(thread):
    tid = ctypes.c_long(thread.ident)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))
    if res == 0:
        print("Invalid thread id")
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        print("PyThreadState_SetAsyncExc failed")

class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode

    def get_response(self, prompt_content):
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }

        out_list = [None]
        time_cal = 0
        t = None
        while out_list[0] == None:
            if time_cal % 60 == 0:
                print("starting new request...")
                if t != None:
                    terminate_thread(t)
                    print("killed current thread")
                t = Thread(target=get_response, args=(self.api_endpoint, payload_explanation, headers, out_list))
                t.start()
            time_cal += 1
            time.sleep(0.5)
        response = out_list[0]

        return response
    # def get_response(self, prompt_content):
    #     BASE_URL = "https://api.chatgptid.net/v1"
    #     OPENAI_API_KEY = "sk-oztvw2gifcNPZQ4bF2E59eDa1b4e4dA5AeE3B562Ce3fA4Dd"
    #     client = openai.OpenAI(
    #         api_key=OPENAI_API_KEY,
    #         base_url=BASE_URL,
    #     )
    #     response = client.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             temperature = 0,
    #             max_tokens = 500,
    #             top_p = 1,
    #             frequency_penalty = 0,
    #             presence_penalty = 0,
    #             messages=[
    #                 {"role": "user", "content": prompt_content}
    #             ]
    #     )

    #     return response.choices[0].message.content