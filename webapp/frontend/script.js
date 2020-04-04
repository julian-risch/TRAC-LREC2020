var explanation = null
var labelMapping = null

$( document ).ready(function() {
        $(document).ajaxStart(function () {
            $("#loading").show()
        }).ajaxStop(function () {
            $("#loading").hide()
        })

	$.ajax({
		url: "/label_mapping",
		method: "GET",
		dataType: 'json',
		contentType: 'application/json; charset=utf-8',
		success: function(data) {
			labelMapping = data
			setLabelMapping(labelMapping)
		},
		error: function(error) {
			alert('error')
		}
	})

        $('#btn-download').on('click', function() {
            explanationCanvas = $('#explanation-div')
            html2canvas(explanationCanvas[0]).then( function(canvas) {
                data = canvas.toDataURL('image/png')
                data = data.replace('/^data:image\/png/', 'data:application/octet-stream')
                download(data, 'explanation.png', '/image/png')
            })
        })
	
	$('#choose-classifier').children().each(function(){
	    $(this).click(function() {
			lime = $('input[type=radio][name=explainability-method][value=lime]')
			lrp = $('input[type=radio][name=explainability-method][value=lrp]')
			prob = $('input[type=radio][name=explainability-method][value=prob]')
			attention = $('input[type=radio][name=explainability-method][value=attention]')
			if (this.value == 'lstm' || this.value == 'svm') {
				lime.prop('disabled', false)
				lrp.prop('disabled', false)
				prob.prop('disabled', true)
				prob.prop('checked', false)
				attention.prop('disabled', true)
				attention.prop('checked', false)
			} else if(this.value == 'naivebayes') {
				lime.prop('disabled', true)
				lime.prop('checked', false)
				lrp.prop('disabled', true)
				lrp.prop('checked', false)
				prob.prop('disabled', false)
				prob.prop('checked', true)
				attention.prop('disabled', true)
				attention.prop('checked', false)
			} else if(this.value == 'att_lstm') {
				lime.prop('disabled', true)
				lime.prop('checked', false)
				lrp.prop('disabled', true)
				lrp.prop('checked', false)
				prob.prop('disabled', true)
				prob.prop('checked', false)
				attention.prop('disabled', false)
				attention.prop('checked', true)
			}
	    })
	})
	
	$('#explain-form').on('submit', function(e){
		e.preventDefault()
		options = getOptions()
		$.ajax({
			url: "/",
	        method: "POST",
	        data: JSON.stringify(options),
			dataType: 'json',
			contentType: 'application/json; charset=utf-8',
			success: function(data) {
				explanation = data
				visParam = parseFloat($('#vis-param-input').val())
				explanationVisualization = renderExplanation(data[0], x => Math.pow(x,visParam))
				$('#explanation-div').empty().append($(explanationVisualization).clone())
				$('#classification').empty().html('Classified as: ' + labelMapping[0][data[1]])
			},
	        error: function(error) {
				alert('error')
	        }
		})
	})
	
	$('#vis-param-input').change(function(){
		renderExplanationStateful()
	})
})

function setLabelMapping(labelMapping) {
	chooseClass = $('#choose-class')
	for (label = 0; label < labelMapping.length; label++) {
		chooseClass.append(new Option('Label: ' + label, label + ' ' + '-1'));
		for (const [index, className] of labelMapping[label].entries()) {
			chooseClass.append(new Option('Label: ' + label + ', Class:' + className, label + ' ' + index));
		}
	}
}

function getLabelAndClass(stringRepresentation) {
	splitted = stringRepresentation.split(' ')
	splitted[0] = parseInt(splitted[0])
	if (splitted[1] == '-1') {
		splitted[1] = null
	} else {
		splitted[1] = parseInt(splitted[1])
	}
	return splitted
}


function renderExplanationStateful() {
	visParam = parseFloat($('#vis-param-input').val())
	explanationVisualization = renderExplanation(explanation[0], x => Math.pow(x,visParam))
	$('#explanation-div').empty().append($(explanationVisualization).clone())
}

function getOptions() {
	options = {}
	options.text = $('#text').val()
	labelAndClass = getLabelAndClass($('#choose-class').val())
	options.label = labelAndClass[0]
	options.class_to_explain = labelAndClass[1] 
	options.classifier = $('input[name=classifier]:checked').val()
	options.options = {}
	if (options.classifier == 'lstm') {
		options.method = $('input[name=explainability-method]:checked').val()
	} else if (options.classifier == 'svm') {
		options.method = $('input[name=explainability-method]:checked').val()
	}
	// Lime Options
	if (options.method == 'lime') {
		options.options.sample_size = parseInt($('#sample-size-input').val())
		options.options.distance_metric = $('#distance-function-input').val()
		options.options.kernel_width = parseInt($('#kernel-width-input').val())
	}
	if (options.method == 'lrp') {
		options.options.eps = parseFloat($('#eps-input').val())
		options.options.bias_factor = parseFloat($('#bias-factor-input').val())
	}
	return options
}

/**
 * Create a div element, that contains a visualization of an explanation, by setting the background color of a token according to the relevance.
 *
 * @param {Explanation} explanation - A list of tokens objects
 * @param {function} [scale] - A function with domain [0,1 ]and range [0,1] applied to the normalized relevance, that returns the intensity of the background color. E.g. to emphasize smaller values. Default is the identity function.
 */
function renderExplanation(explanation, scale = x => x) {
    
    // The div container to be filled with the visualization and returned
    container = document.createElement('div')
    
    // Get the maximum absolute relevance, to normalize the relevances 
    maxAbsoluteRelevance = 0
    explanation.forEach(function(token){
        if (token.hasOwnProperty('relevance')) {
            absoluteRelevance = Math.abs(token['relevance'])
            maxAbsoluteRelevance = absoluteRelevance > maxAbsoluteRelevance ? absoluteRelevance : maxAbsoluteRelevance
        }
    })

    // Create a span for every token and set the background color to represent the relevance
    explanation.forEach(function(token){
        word = token['token']
        if (word == '\n') {
            container.appendChild(document.createElement('br'))
        } else {
            node = document.createElement('span')
			node.setAttribute('class', 'token')
			tokenTooltip = document.createElement('span')
            tokenTooltip.setAttribute('class', 'tokentooltip')
            if (token.hasOwnProperty('relevance')) {
                relevance = token['relevance']
                if (relevance >= 0) {
                	intensity = scale(relevance / maxAbsoluteRelevance)
                    color = 'rgba(255, 0, 0, ' + intensity + ')'
                } else {
                	intensity = scale(-relevance / maxAbsoluteRelevance)
                    color = 'rgba(0, 0, 255, ' + intensity + ')'
                }
				tokenTooltip.appendChild(document.createTextNode(relevance))
            } else {
				color = 'rgba(0,0,0,0)'
				tokenTooltip.appendChild(document.createTextNode('Ignored'))
			}
            node.setAttribute('style', 'background-color: ' + color)
            node.appendChild(document.createTextNode(word))
			node.appendChild(tokenTooltip)
            container.appendChild(node)
        }
    })
    return container
} 
