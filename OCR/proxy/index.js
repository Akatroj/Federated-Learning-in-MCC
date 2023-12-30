const express = require("express");

const app = express();
app.use(express.json({ limit: "50mb" }));

const proxiedUrl = "http://0.0.0.0:8080/base64";
// const proxiedUrl = "http://172.18.0.3:31555/base64";

function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random(); // Converting [0,1) to (0,1]
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  // Transform to the desired mean and standard deviation:
  return z * stdev + mean;
}

const SHOULD_DELAY = true;

const delayTransmission = async () => {
  return new Promise((res, _) => {
    setTimeout(() => {
      res();
    }, Math.max(0, gaussianRandom(300, 100)));
  });
};

app.post("/base64", async (request, response) => {
  console.log("got request");
  if (SHOULD_DELAY) {
    await delayTransmission();
  }

  const start = new Date();
  try {
    const resp = await fetch(proxiedUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request.body),
    });

    if (!resp.ok) {
      response.status(resp.status).send();
    } else {
      const respData = await resp.json();
      respData.computationTimeMillis = new Date().getTime() - start.getTime();
      response.status(200).json(respData);
    }
  } catch {
    response.status(500).send();
  }
});

app.listen(8081, () => {
  console.log("Listenning on :8081");
});
