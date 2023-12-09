const express = require("express");
const { exec } = require("child_process");

const app = express();
app.use(express.json());

app.get("/info", (request, response) => {
  try {
    exec(
      'kubectl get --raw "/apis/metrics.k8s.io/v1beta1/nodes"',
      (err, stdout, stderr) => {
        if (err) {
          // node couldn't execute the command
          return;
        }

        const res = JSON.parse(stdout);
        console.log(res);

        // the *entire* stdout and stderr (buffered)
        console.log(`stdout: ${stdout}`);
        console.log(`stderr: ${stderr}`);
        response.send({ nodeNumber: res.items.length });
      }
    );
  } catch (error) {
    res.status(500).send("Something gone wrong");
  }
});

app.listen(8080, () => {
  console.log("Start listen on port 8080");
});
