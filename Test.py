# This file can be used for testing your code but I would recommend just using postman

import requests

print(
    requests.get(
        "http://127.0.0.1:8000/get-message",
        #json={"trans_date_trans_time" = somet, "cc_num" = somet, etc.}
      ).json())