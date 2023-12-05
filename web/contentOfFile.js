const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const server = http.createServer((req, res) => {
    const reqUrl = url.parse(req.url, true);
    const pathname = reqUrl.pathname === '/' ? '/index.html' : reqUrl.pathname;

    if (req.method === 'GET') {
        if (pathname === '/index.html') {
            // Serve HTML page with input field
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(`
        <!DOCTYPE html>
        <html>
          <head>
            <title>File Reader</title>
          </head>
          <body>
            <h1>Enter filename:</h1>
            <form action="/" method="POST">
              <input type="text" name="filename" />
              <input type="submit" value="Read File" />
            </form>
          </body>
        </html>
      `);
        } else {
            // Handle other requests
            res.writeHead(404, { 'Content-Type': 'text/plain' });
            res.end('Not Found');
        }
    } else if (req.method === 'POST' && reqUrl.pathname === '/') {
        // Handle form submission
        let body = '';
        req.on('data', (chunk) => {
            body += chunk;
        });

        req.on('end', () => {
            const filename = decodeURIComponent(body.split('=')[1]);
            const filePath = path.join(__dirname, filename);

            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    res.writeHead(404, { 'Content-Type': 'text/plain' });
                    res.end('File not found');
                } else {
                    res.writeHead(200, { 'Content-Type': 'text/html' });
                    res.end(`
            <!DOCTYPE html>
            <html>
              <head>
                <title>File Reader</title>
              </head>
              <body>
                <h1>File Content:</h1>
                <pre>${data}</pre>
              </body>
            </html>
          `);
                }
            });
        });
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
