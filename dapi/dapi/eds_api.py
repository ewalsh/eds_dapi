import requests
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


def mapToDict(dat: List) -> Dict:
    return {dat[0]['data'][i][0]: dat[0]['data'][i][1] for i in range(0, len(dat[0]['data']))}


class EDSsession():
    # init function
    def __init__(self):
        eds_email = os.getenv('EDS_API_EMAIL', 'secret')
        eds_pass = os.getenv('EDS_API_PASSWORD', 'secret')
        self.session = requests.Session()
        self.payload = {'email': eds_email, 'password': eds_pass}
        self.session.post(
            'https://economicdatasciences.com/login', data=self.payload)

    # benchmark
    def get_benchmark(self, code):
        get_str = 'https://economicdatasciences.com/benchmarks/' + code
        res = self.session.get(get_str)
        return(res.json())
