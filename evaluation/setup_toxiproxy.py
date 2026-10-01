"""Create the test-only PostgreSQL and Redis Toxiproxy routes."""

import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROXIES = {
    "postgres": {"listen": "0.0.0.0:15432", "upstream": "postgres:5432"},
    "redis": {"listen": "0.0.0.0:16379", "upstream": "redis:6379"},
}


class ToxiproxyClient:
    def __init__(self, base_url="http://localhost:8474"):
        self.base_url = base_url.rstrip("/")

    def request(self, method, path, payload=None, expected=(200, 201, 204)):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=5) as response:
                body = response.read()
                if response.status not in expected:
                    raise RuntimeError(f"Unexpected Toxiproxy status: {response.status}")
                return json.loads(body) if body else None
        except HTTPError as error:
            if error.code not in expected:
                raise
            body = error.read()
            return json.loads(body) if body else None

    def ensure_proxy(self, name, listen, upstream):
        try:
            current = self.request("GET", f"/proxies/{name}")
        except HTTPError as error:
            if error.code != 404:
                raise
            return self.request("POST", "/proxies", {
                "name": name,
                "listen": listen,
                "upstream": upstream,
                "enabled": True,
            })
        return self.request("POST", f"/proxies/{name}", {
            "listen": listen,
            "upstream": upstream,
            "enabled": True,
        })

    def set_enabled(self, name, enabled):
        return self.request("POST", f"/proxies/{name}", {"enabled": enabled})

    def add_toxic(self, proxy, name, toxic_type, attributes, stream="downstream"):
        return self.request("POST", f"/proxies/{proxy}/toxics", {
            "name": name,
            "type": toxic_type,
            "stream": stream,
            "toxicity": 1.0,
            "attributes": attributes,
        })

    def clear_toxics(self, proxy):
        for toxic in self.request("GET", f"/proxies/{proxy}/toxics") or []:
            self.request(
                "DELETE",
                f"/proxies/{proxy}/toxics/{toxic['name']}",
                expected=(204,),
            )


def configure(client):
    for name, values in PROXIES.items():
        client.ensure_proxy(name, **values)
        client.clear_toxics(name)
        client.set_enabled(name, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8474")
    args = parser.parse_args()
    configure(ToxiproxyClient(args.url))
    print("Toxiproxy routes ready: PostgreSQL :15432, Redis :16379")


if __name__ == "__main__":
    main()
