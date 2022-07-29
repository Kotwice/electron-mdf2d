document.addEventListener('DOMContentLoaded', (event) => {
    console.log(event)

    let socket = io('ws://localhost:4000')

    function isValidJsonString(jsonString) {    
        if(!(jsonString && typeof jsonString === "string")){
            return false;
        }    
        try{
           JSON.parse(jsonString);
           return true;
        }catch(error){
            return false;
        }    
    }

    let inputs = {
        x: {id: 'input-x', val: [0, 250e-3, 10], limits: [0, 1000], length: 3},
        y: {id: 'input-y', val: [0, 50e-3, 10], limits: [0, 1000], length: 3},
        t: {id: 'input-t', val: [0, 0.01, 10], limits: [0, 1000], length: 3},
        rho: {id: 'input-rho', val: [1], limits: [0, 1000], length: 1},
        nu: {id: 'input-nu', val: [1], limits: [0, 1000], length: 1},
        iter_res: {id: 'input-iter-res', val: [500], limits: [0, 10000], length: 1},
        tol: {id: 'input-tol', val: [1e-2], limits: [0, 1], length: 1},
        iter_ref: {id: 'input-iter-ref', val: [10], limits: [0, 1000], length: 1},
    }

    function checkValues (id, limits, length) {
        let flag
        let line = $('#' + id).val()    
        if (isValidJsonString(line)) {
            let object = JSON.parse(line)    
            if (object.length == length) {
                for (let index in object) {
                    let value = object[index]
                    flag = (!isNaN(value)) ? ((value >= limits[0] && value <= limits[1]) ? true : false) : false
                }
            }
            else {
                flag = false;
            }
        }    
        else {
            flag = false;
        }
        flag ? $('#' + id).removeClass('is-invalid').addClass('is-valid') : $('#' + id).removeClass('is-valid').addClass('is-invalid')
        
        let flag_global = false

        for (let key in inputs) {
            if (document.getElementById(inputs[key]['id']).classList.contains('is-invalid')) {
                flag_global = true
            }
        }

        flag_global ? $('#button-calculate').prop('disabled', true) : $('#button-calculate').prop('disabled', false)
    }

    for (let key in inputs) {
        $('#' + inputs[key]['id']).val(JSON.stringify(inputs[key]['val']))
        let input = inputs[key]
        checkValues(input['id'], input['limits'], input['length'])
        $('#' + input['id']).on('blur', () => {checkValues (input['id'], input['limits'], input['length'])})
    }

    $('#button-calculate').on('click', () => {
        let data = {}
        for (let key in inputs) {
            data[key] = JSON.parse($('#' + inputs[key]['id']).val())
        }
        console.log(data)
        socket.emit('calculate', data)
        $('#flush-collapse-process').collapse('show')
        layout = {xaxis: {type: 'log', autorange: true, title: 'iterations'}, yaxis: {type: 'log', autorange: true, title: 'error'},
            title: {text: 'Convergence Plot'}}
        Plotly.newPlot('figure-process', [{y: [0], mode: 'lines', type: 'scatter'}], layout)
    })

    socket.on('plot-figure-process', event => {
        Plotly.extendTraces('figure-process', {y: [[event['tolerance']]]}, [0])
    })

    $('#flush-collapse-parameter').collapse('show')

    socket.on('plot-figure-result', figure => {
        console.log(figure)
        $('#flush-collapse-result').collapse('show')
        Plotly.newPlot('figure-result', figure['data'], figure['layout'], figure['frames'])
    })

})