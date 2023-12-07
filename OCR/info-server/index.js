const express = require("express");
const { exec } = require("child_process");

const app = express();
app.use(express.json());

app.get("/info", (request, response) => {
  try {
    exec("kubectl top node", (err, stdout, stderr) => {
      if (err) {
        // node couldn't execute the command
        return;
      }
      const lines = stdout.trim().split("\n").slice(1);
      const numberOfNodes = lines.length;

      const cpuPercentages = [];
      const memoryPercentages = [];
      lines.forEach((line) => {
        const values = line.split(/\s+/);
        cpuPercentages.push(
          Number.parseInt(values[2].slice(0, values[2].length))
        );
        memoryPercentages.push(
          Number.parseInt(values[4].slice(0, values[4].length))
        );
      });

      const computeResourceStats = (resourcePrefix, resourceInfos) => {
        const totalUsage = resourceInfos.reduce((acc, cur) => acc + cur, 0);
        const meanUsage = totalUsage / resourceInfos.length;
        const variance =
          resourceInfos.length <= 1
            ? 0
            : resourceInfos.reduce(
                (acc, cur) => acc + Math.pow(cur - meanUsage, 2),
                0
              ) /
              (resourceInfos.length - 1);

        return {
          [`${resourcePrefix}Min`]: Math.min(...resourceInfos),
          [`${resourcePrefix}Max`]: Math.max(...resourceInfos),
          [`${resourcePrefix}Mean`]: meanUsage,
          [`${resourcePrefix}Var`]: variance,
        };
      };

      const stats = {
        ...computeResourceStats("cpu", cpuPercentages),
        ...computeResourceStats("mem", memoryPercentages),
        numberOfNodes: numberOfNodes,
      };

      // the *entire* stdout and stderr (buffered)
      console.log(`stdout: ${stdout}`);
      console.log(`stderr: ${stderr}`);
      response.send(stats);
    });
  } catch (error) {
    res.status(500).send("Something gone wrong");
  }
});

app.listen(8080, () => {
  console.log("Start listen on port 8080");
});
