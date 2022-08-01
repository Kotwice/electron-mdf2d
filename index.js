const {app, BrowserWindow} = require('electron')

const io = require('socket.io')(4000)

let os = require('os');

app.disableHardwareAcceleration()

var spawn = require("child_process").spawn

io.on('connection', client => {

	client.on('calculate', (event) => {
		let tast = spawn('python', ["./resources/scripts/calculate_test.py", JSON.stringify(event)])
		// let tast = spawn('python', ["./resources/scripts/calculate_task.py", JSON.stringify(event)])

		tast.stderr.on('data', data => {
			console.error(`stderr: ${data}`);
		});
	})

	client.on('figure-process', data => {
		client.broadcast.emit('plot-figure-process', JSON.parse(data))
	})

	client.on('figure-result', data => {
		client.broadcast.emit('plot-figure-result', JSON.parse(data))
	})

	// client.on('get-state', () => {
	// 	console.log(os.cpus())
	// 	client.broadcast.emit('res-state', {cpu: os.cpus()})
	// })

});

app.whenReady().then(() => {

    const window = new BrowserWindow (
		{
			width: 1200,
			height: 700,
			frame: true,
			title: 'MDF2D',
			webPreferences: {
				nodeIntegration: true
			}
		}
    )

    window.webContents.openDevTools()

    window.loadFile('./resources/pages/index.html')
  
})
  
app.on('window-all-closed', () => {

	if (process.platform !== 'darwin') {

		app.quit()

	}

})