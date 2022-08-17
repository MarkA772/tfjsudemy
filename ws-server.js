const io = require("socket.io");

const server = new io.Server({
  cors: {
    origin: "http://localhost:8080"
  }
});

server.on("connection", (socket) => {
  console.log("user connected");
  socket.on('loss message', (arg) => {
    console.log(arg);
  });
});

server.listen(3333);


