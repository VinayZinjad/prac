const http = require("http");
const fs = require("fs");
const url = require("url");
const path = require("path");

const server = http.createServer((req, res) => {
    const { pathname, query } = url.parse(req.url, true);

    if (pathname == "/") {
        //Server HTML

        res.writeHead(200, { "Content-Type": "text/html" });
        fs.createReadStream("append-files.html").pipe(res);
    } else if (pathname === "/appendFiles" && req.method === "POST") {
        let body = "";

        req.on("data", (chunk) => {
            body += chunk;
        });

        req.on("end", () => {
            const formData = new URLSearchParams(body);
            const firstFileName = formData.get("firstFileName");
            const secondFileName = formData.get("secondFileName");

            // read the content

            fs.readFile(firstFileName, "utf-8", (err, data) => {
                if (err) {
                    res.writeHead(500, { ContentType: "text/plain" });
                    res.end("Error reading files.");
                } else {
                    fs.appendFile(secondFileName, data, (err) => {
                        if (err) {
                            res.writeHead(500, { ContentType: "text/plain" });
                            res.end("Error reading files.");
                        } else {
                            res.writeHead(200, { ContentType: "text/plain" });
                            res.end("updated.");
                        }
                    });
                }
            });
        });
    }
});
const port = 8081;
server.listen(port, () => {
    console.log(`server listening on ${port}`);
});