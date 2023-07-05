var annotation = {};
var N_QUALS = 1;
var corrected_pred_list = [];
var preds_ranks_list = []
/*! js-cookie v3.0.0-beta.3 | MIT */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?module.exports=t():"function"==typeof define&&define.amd?define(t):(e=e||self,function(){var n=e.Cookies,r=e.Cookies=t();r.noConflict=function(){return e.Cookies=n,r}}())}(this,function(){"use strict";var e={read:function(e){return e.replace(/(%[\dA-F]{2})+/gi,decodeURIComponent)},write:function(e){return encodeURIComponent(e).replace(/%(2[346BF]|3[AC-F]|40|5[BDE]|60|7[BCD])/g,decodeURIComponent)}};function t(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)e[r]=n[r]}return e}return function n(r,o){function i(e,n,i){if("undefined"!=typeof document){"number"==typeof(i=t({},o,i)).expires&&(i.expires=new Date(Date.now()+864e5*i.expires)),i.expires&&(i.expires=i.expires.toUTCString()),n=r.write(n,e),e=encodeURIComponent(e).replace(/%(2[346B]|5E|60|7C)/g,decodeURIComponent).replace(/[()]/g,escape);var c="";for(var u in i)i[u]&&(c+="; "+u,!0!==i[u]&&(c+="="+i[u].split(";")[0]));return document.cookie=e+"="+n+c}}return Object.create({set:i,get:function(t){if("undefined"!=typeof document&&(!arguments.length||t)){for(var n=document.cookie?document.cookie.split("; "):[],o={},i=0;i<n.length;i++){var c=n[i].split("="),u=c.slice(1).join("=");'"'===u[0]&&(u=u.slice(1,-1));try{var f=e.read(c[0]);if(o[f]=r.read(u,f),t===f)break}catch(e){}}return t?o[t]:o}},remove:function(e,n){i(e,"",t({},n,{expires:-1}))},withAttributes:function(e){return n(this.converter,t({},this.attributes,e))},withConverter:function(e){return n(t({},this.converter,e),this.attributes)}},{attributes:{value:Object.freeze(o)},converter:{value:Object.freeze(r)}})}(e,{path:"/"})});

/**
 * Map from selection id (e.g., "selection-0") to the Characters
 */
var error_types_dict = {
    "Missing-Answer": "Missing Answer",
    "Missing-Major-Auxiliary": "Missing Major Auxiliary Info",
    "Missing-Minor-Auxiliary": "Missing Minor Auxiliary Info",
    "Wrong-Grounding" : "Inconsistent",
    "Unverifiable" : "Unverifiable",
    "Irrelevant" : "Irrelevant",
    "Redundant" : "Repetitive",
    "Incoherent" : "Incoherent"
};
var situation_text = {};
var passageId2nSents = [];
var prediction_btn_checked = [];
var cur_task_idx = 0;
var isComparison = false;


var colors = ["DarkGreen", "DarkBlue", "Orange", "Brown"]

function substitute(input_text) {
    let new_input_text = input_text.replace(/,/g, "_SEP_");
    new_input_text = new_input_text.replace(/"/g, "_QUOTE_");
    new_input_text = new_input_text.replace(/</g, "_LEFT_");
    new_input_text = new_input_text.replace(/>/g, "_RIGHT_");
    new_input_text = new_input_text.replace(/[\r\n]+/g, "_NEWLINE_");
    return new_input_text
}

var ALLOW_TO_SUBMIT = "<p style='color:Blue'><br/><em><big>You are allowed to submit now!</big></em></p>";

var NO_EXPECTED_ERROR_LABEL = "Error: You should either check the no error checkbox, or mark errors in STEP 1."
var NO_EXPECTED_CORRECTION = "Error: If you do not check the no error checkbox, the correction box in STEP 2 should be changed."
var UNEXPECTED_CORRECTION = "Error: If you check the no error checkbox, the correction box should be unchanged (use the reset button). Otherwise, you should mark errors in STEP 1."
var UNREAD_PREDICTIONS = "Error: You can not submit without reading all 6 model prediction pairs and properly comparing them."

var MISSING_OR_INVALID_PID_SID = 'Error: The passage id and sentence id(s) are invalid, for verifying a <em>SINGLE</em> piece of information.'
var MISSING_OR_INVALID_ANTECEDANT = 'Error: An antecedant span should be a subtring from the text before your selected span.'

var ALL_SAME_WARNING = "<p style='color:Blue'>Warning: You indicate all model predications are equally good, which is rare. Make sure you read our instruction and all model predictions carefully.</p>"
var CHECK_EVERY_PASSAGE_WARNING = "<p style='color:Blue'>Warning: Make sure you check every passage before making the decision that this information is unverifiable.</p>"
var MINIMAL_ANTECEDANT_SPAN_WARNING = "<p style='color:Blue'>Warning: Make sure you input a <em>minimal</em> antecedant span that is being repeated.</p>"

var situationID2Idx2Selected = {};

/**
 * All selected spans
 */
class Characters {
    constructor(situationID, num) {
        this.situationID = situationID;
        annotation[situationID] = {};
        this.data = [];
        this.displayID = situationID + '-display';
    }
    add(newCS) {
        // check for duplicates and add if it's not there.
        for (let oldCS of this.data) {
            if (oldCS == null) {
                continue;
            }
            if (oldCS.equals(newCS)) {
                // animate it to show it exists.
                oldCS.noticeMeSenpai = true;
                return;
            }
        }
        this.data.push(newCS);
        if (this.situationID.startsWith('situation')) {
            for (let i = newCS.start_end_pairs[0][0]; i < newCS.start_end_pairs[0][1]; ++i) {
                situationID2Idx2Selected[this.situationID][i] = true
            }
        }
    }
    remove(cs) {
        for (let i = this.data.length - 1; i >= 0; i--) {
            if (this.data[i] == null) {
                continue;
            }
            if (this.data[i].equals(cs)) {
                this.data[i] = null
            }
        }
        this.data = this.data.filter(v => v !== null)
        if (this.situationID.startsWith('situation')) {
            for (let i = cs.start_end_pairs[0][0]; i < cs.start_end_pairs[0][1]; ++i) {
                situationID2Idx2Selected[this.situationID][i] = false
            }
        }
    }
    update() {
        this.render();
        this.serialize();
    }
    render() {
        let display = $('#' + this.displayID).empty();
        for (let i = 0; i < this.data.length; i++) {
            if (this.data[i] == null) {
                continue;
            }
            display.append(this.data[i].render(this.situationID, i));
        }
    }
    serialize() {
        let strings = [];
        for (let character of this.data) {
            if (character == null) {
                continue;
            }
            strings.push(character.serialize());
        }
        if (strings.length > 0) {
            if (this.situationID.startsWith("situation-")) {
                annotation[this.situationID] = {"spans": strings, "text": situation_text[this.situationID]};
            } else {
                annotation[this.situationID] = {"spans": strings}
            }
        }
        else {
            annotation[this.situationID] = {};
        }
    }
}
class CharacterSelection {
    constructor(error_type, explanation, start_end_pairs, num) {
        this.error_type = error_type;
        this.explanation = explanation;
        this.start_end_pairs = start_end_pairs
        this.num = num
        this.noticeMeSenpai = false;
    }
    equals(other) {
        return this.error_type == other.error_type && this.explanation == other.explanation
                && JSON.stringify(this.start_end_pairs) === JSON.stringify(other.start_end_pairs);
    }
    render(situationID, num) {
        let error_type = this.error_type, explanation = this.explanation, start_end_pairs = this.start_end_pairs; // so they go in the closure
        let txt = error_types_dict[error_type];
        if ($.trim(explanation).length > 0) {
            txt = error_types_dict[error_type] + ": [explanation] " + explanation;
        }
        
        let color_class= error_type
        let text_color = "white"
        let opposite_color = "black"

        let removeButton = $('<button></button>')
            .addClass('bg-transparent ' + text_color +' bn hover-' + opposite_color + ' hover-bg-' + text_color + ' br-pill mr1 pointer')
            .append('✘')
            .on('click', function () {
                if (situationID.startsWith("situation-")) {
                    document.getElementById(situationID).innerHTML = situation_text[situationID]
                }
                C[situationID].remove(new CharacterSelection(error_type, explanation, start_end_pairs));
                if (situationID.startsWith("situation-")) {
                    annotate(C[situationID], situation_text[situationID], situationID)
                }
                C[situationID].update();
                getAnnotations();
            });

        let span = $('<span></span>')
            .addClass('b bg-' + color_class + " " + text_color +' pa2 ma1 br-pill dib quality-span')
            .append(removeButton)
            .append(txt);
        span.attr('id', 'quality-span-'+num)
        span.attr('data-situation-id', situationID)
        span.attr('data-error-type', error_type)
        span.attr('data-explanation', explanation)
        span.attr('data-start-end-pairs', start_end_pairs)
        span.attr('data-num', num)
        // if the character needs to be noticed, abide.
        if (this.noticeMeSenpai) {
            this.noticeMeSenpai = false;
            span.addClass("animated bounce faster");
            setTimeout(function () {
                span.removeClass('animated bounce faster');
            }, 1000);
        }
        return span;
    }
    serialize() {
        var serialized = {};
        serialized["error type"] = substitute(this.error_type)
        serialized["explanation"] = this.explanation
        serialized["start"] = this.start_end_pairs[0][0]
        serialized["end"] = this.start_end_pairs[0][1]
        return JSON.stringify(serialized);
    }
}

/**
 * Span highlighting functions
 */

// globals
let C = {}
let start_end_pairs = []
let situationID;

function comparespan(span_a, span_b) {
    let index_a = span_a[1]
    let index_b = span_b[1]
    if(index_a == index_b) {
        return span_a[3] - span_b[3]
    }
    return index_a - index_b;
}

function annotate(character, text, situationID) {
    let character_selections = character.data

    let span_list = []
    for(selection of character_selections) {
        if (selection == null) {
            continue;
        }
        let num = selection.num
        let p_span_id = "p-span-" + num + '-' + situationID
        let start_end_pair = selection.start_end_pairs[0]
        span_list.push([p_span_id, start_end_pair[0], true, num, selection.error_type]);
        span_list.push([p_span_id, start_end_pair[1], false, num, selection.error_type]);
    }

    // console
    console.log(span_list)
    span_list.sort(comparespan)

    let new_text = ""
    for(i in span_list) {
        span = span_list[i]
        var before_pair_end;
        if(i == 0) {
            before_pair_end = 0
        } else{
            before_pair_end = span_list[i - 1][1]
        }
        start_temp = span[1]
        subtxt = text.substring(before_pair_end, start_temp)
        var span_to_add;
        var color_class = span[4]

        if(span[2]) {
            span_to_add = "<span class=\"annotation border-" + color_class + " " + span[0]+ "\">"
        } else {
            span_to_add = "</span>"
            // multiple spans cross together (intersect)
            for (j = i; j >0; j--) {
                if (span_list[j - 1][2] && span_list[j-1][3] != span[3]) {
                    var previous_color_class = span_list[j-1][4]
                    span_to_add += "</span>"
                } else {
                    break
                }
            }
            for (j = i; j >0; j--) {
                if (span_list[j - 1][2] && span_list[j-1][3] != span[3]) {
                    var previous_color_class = span_list[j-1][4]
                    span_to_add += "<span class=\"annotation border-" + previous_color_class + " p-span-" + span_list[j-1][3]+ "\">"
                } else {
                    break
                }
            }
        }
        new_text += subtxt + span_to_add
    }
    if (span_list.length == 0) {
        new_text += text
    } else {
        new_text += text.substring(span_list[span_list.length - 1][1])
    }
    document.getElementById(situationID).innerHTML = new_text
};

function annotate_select_span(character, text, select_span, situationID) {
    let character_selections = character.data
    let span_list = []
    for(selection of character_selections) {
        if (selection == null) {
            continue;
        }
        let num = selection.num
        let p_span_id = "p-span-" + num + '-' + situationID
        let start_end_pair = selection.start_end_pairs[0]
        span_list.push([p_span_id, start_end_pair[0], true, num, selection.error_type]);
        span_list.push([p_span_id, start_end_pair[1], false, num, selection.error_type]);
    }

    if (select_span !== undefined) {
        span_list.push(["select-span--1", select_span[0], true, -1, "select-span"]);
        span_list.push(["select-span--1", select_span[1], false, -1, "select-span"]);
    }
    span_list.sort(comparespan)
    let new_text = ""
    for(i in span_list) {
        span = span_list[i]
        var before_pair_end;
        if(i == 0) {
            before_pair_end = 0
        } else{
            before_pair_end = span_list[i - 1][1]
        }
        start_temp = span[1]
        subtxt = text.substring(before_pair_end, start_temp)
        var span_to_add;
        var color_class = span[4]
        if(span[2]) {
            span_to_add = "<span class=\"annotation border-" + color_class + " " + span[0]+ "\">"
            if (span[4] == "select-span") {
                span_to_add = "<span class=\"annotation bg-yellow " + span[0]+ "\">"
            }
        } else {
            span_to_add = "</span>"
            // multiple spans cross together (intersect)
            for (j = i; j >0; j--) {
                if (span_list[j - 1][2] && span_list[j-1][3] != span[3]) {
                    var previous_color_class = span_list[j-1][4]
                    span_to_add += "</span>"
                } else {
                    break
                }
            }
            for (j = i; j >0; j--) {
                if (span_list[j - 1][2] && span_list[j-1][3] != span[3]) {
                    var previous_color_class = span_list[j-1][4]
                    if (span_list[j - 1][4] == "select-span") {
                        span_to_add += "<span class=\"annotation bg-yellow " + span_list[j-1][0] + "\">"
                    }
                    span_to_add += "<span class=\"annotation border-" + previous_color_class + " " + span_list[j-1][0]+ "\">"
                } else {
                    break
                }
            }
        }
        new_text += subtxt + span_to_add
    }
    if (span_list.length == 0) {
        new_text += text
    } else {
        new_text += text.substring(span_list[span_list.length - 1][1])
    }
    document.getElementById(situationID).innerHTML = new_text
};


/**
 ***********************************  Main Logic Starts here  ***********************************
 */


function reset_redundant_grounding_explanation() {
    $('#missing-info-explanation-passage').val('');
    $('#missing-info-explanation-sent').val('');
    $('#intrinsic-explanation-passage').val('');
    $('#intrinsic-explanation-sent').val('');
    $('#redundant-explanation').val('');
    $('#missing_info_explanation_div').hide();
    $('#grounding_explanation_div').hide();
    $('#redundant_explanation_div').hide();
}

function disable_everything() {
    $("input[name='error_type']").removeClass("selected")
    $("input:radio[name='error_type']").prop('checked', false);

    reset_redundant_grounding_explanation();
    $("#button_div").addClass("disable");
    document.getElementById('error_selection_warning_div').innerHTML = '';
}

function qualButtonClicked(button_id) {
    getAnnotations();
    
    cur_task_idx = parseInt(button_id.substring(button_id.length-1));
  
    $(".qual-hits").hide();
    $(".hit-" + String(cur_task_idx)).show();
    disable_everything();

    $('.qual-hits-btn').css("border", "0px");
    $("#btn-hit-" + String(cur_task_idx)).css("border", "black 3px solid");
  
    getAnnotations();
}

function addEvidence(evidence, idx) {
    for (var i = 0; i < evidence.length; ++i) {
        
        evidenceDiv = $(`<div class="pl4 pr4">
                    <p>
                        <span class="passage">Passage ` + String(i+1) + `:</span> 
                        <span class="passage_title"><em>Title (S0)</em> - ` + evidence[i]['wikipage'] + ` </span>
                        <button type="button" class="show-hide-btn" id="passage-` + String(i) + `-btn-` + String(idx) + `"> Click to hide </button>
                    </p>
                    </div>`)
        
        var sentences_div_string = `<div id="passage-` + String(i) + `-` + String(idx) + `">`
        for (var j = 0; j < evidence[i]["content"].length; ++j) {
            sentences_div_string += `<div><strong><em>S` + String(j+1) + `.</em></strong> ` + evidence[i]["content"][j] + `</div>`
        }
        
        passageId2nSents[idx][i+1] = evidence[i]["content"].length;
        sentences_div_string += "</div>"

        evidenceDiv.append(sentences_div_string)
        $(`#situation-`+ String(idx) + `-passage-div`).append(evidenceDiv)
        if (i !== evidence.length-1) {
            $(`#situation-` + String(idx) + `-passage-div`).append('<hr>')
        } 
    }
}

function constructPairComprison(parsed_question, pred1_idx, pred2_idx) {
    var pred1_key = "predicted answer" + ((pred1_idx === 1)? '' : ' ' + String(pred1_idx))
    var pred2_key = "predicted answer" + ((pred2_idx === 1)? '' : ' ' + String(pred2_idx))
    var pred1 = parsed_question[pred1_key]
    var pred2 = parsed_question[pred2_key]
    var color1 = colors[pred1_idx-1]
    var color2 = colors[pred2_idx-1]
    return `<hr><strong>Pred #` + String(pred1_idx) + `: </strong><span style="color:` + color1 + `">` + pred1 + `</span><hr><strong>Pred #` + String(pred2_idx) + `: </strong><span style="color:` + color2 + `">` + pred2 + `</span><hr>`;
}

function load(input) {
    // script
    $(document).ready(function () {
        isComparison = $('#taskKey').attr("value")=='"comparison"'

        if (isComparison) {
            $("#situation-0-generation-div").hide()
        } else {
            $("#compare-div-0").hide()
        }

        // Load inputs
        prompt_list = [input]

        
        for (var i = 0; i < prompt_list.length; ++i) {
            var prompt = prompt_list[i]
            var parsed_question = JSON.parse(prompt)

            passageId2nSents.push({})

            situationID2Idx2Selected["situation-" + String(i)] = {}
            
      
            document.getElementById("situation-" + String(i) + "-prompt").innerHTML = parsed_question["question"]; // "prompt";
            document.getElementById("situation-" + String(i) + "-gold").innerHTML = parsed_question["gold answer"]; // "reference";
            
            addEvidence(parsed_question["passages"], i)
            
            document.getElementById("situation-" + String(i)).innerHTML = parsed_question["predicted answer"]; // "prediction";
            $('#correction_inline-' + String(i)).val($('#situation-' + String(i)).text());
            document.getElementById("prediction-1-" + String(i)).innerHTML = constructPairComprison(parsed_question, 1, 2);
            // TODO
            document.getElementById("prediction-2-" + String(i)).innerHTML = constructPairComprison(parsed_question, 1, 3);
            document.getElementById("prediction-3-" + String(i)).innerHTML = constructPairComprison(parsed_question, 1, 4); 
            document.getElementById("prediction-4-" + String(i)).innerHTML = constructPairComprison(parsed_question, 2, 3);
            document.getElementById("prediction-5-" + String(i)).innerHTML = constructPairComprison(parsed_question, 2, 4);
            document.getElementById("prediction-6-" + String(i)).innerHTML = constructPairComprison(parsed_question, 3, 4);

      
            // build up elements we're working with
            situation_text['situation-' + String(i)] = $('#situation-' + String(i)).text()
            C["situation-" + String(i)] = new Characters("situation-" + String(i), 0);
            C["missing-info-" + String(i)] = new Characters("missing-info-" + String(i), 0);

            corrected_pred_list.push('')
            preds_ranks_list.push([])
            prediction_btn_checked.push([false, false, false, false, false, false])
        }
        
        qualButtonClicked("#btn-hit-0");

        // initialize our data structures NOTE: later we'll have to add data that's loaded
        // into the page (the machine's default guesses). or... maybe we won't?
        var pageX;
        var pageY;

        $(".show-hide-btn").click(function () {
            var content_id = this.id.replace('-btn', '')
            if($("#"+content_id).is(":visible")){
                $("#"+content_id).hide();
                $("#"+this.id).html("Click to show");
            } else {
                $("#"+content_id).show();
                $("#"+this.id).html("Click to hide");
            }
            if (String(this.id).startsWith('prediction')) {
                prediction_btn_checked[cur_task_idx][parseInt(content_id.replace('prediction-', ''))-1] = true
            }
            getAnnotations();
        });
    
        $('#close-icon').on("click", function(e) {
            $('#error_type').val('');

            reset_redundant_grounding_explanation();
            $("#quality-selection").fadeOut(0.2);
            start_end_pairs = []
            if (situationID.startsWith('situation-')) {
                annotate(C[situationID], situation_text[situationID], situationID);
            }
            disable_everything();
            getAnnotations();
        });
        $(".situation-text").on("mousedown", function(e){
            pageX = e.pageX;
            pageY = e.pageY;
            document.getElementById(this.id).innerHTML = situation_text[this.id]
        });
        $(".situation-text").on('mouseup', function (e) {

            situationID = e.target.id;
            let selection = window.getSelection();
            if (selection.anchorNode != selection.focusNode || selection.anchorNode == null) {
                // highlight across spans
                return;
            }
            
            let range = selection.getRangeAt(0);
            let [start, end] = [range.startOffset, range.endOffset];
            if (start == end) {
                // disable on single clicks
                annotate(C[situationID], situation_text[situationID], situationID)
                return;
            }
            // manipulate start and end to try to respect word boundaries and remove
            // whitespace.
            end -= 1; // move to inclusive model for these computations.
            let txt = $('#' + situationID).text();
            while (txt.charAt(start) == ' ') {
                start += 1; // remove whitespace
            }
            while (start - 1 >= 0 && txt.charAt(start - 1) != ' ') {
                start -= 1; // find word boundary
            }
            while (txt.charAt(end) == ' ') {
                end -= 1; // remove whitespace
            }
            while (end + 1 <= txt.length - 1 && txt.charAt(end + 1) != ' ') {
                end += 1; // find word boundary
            }
            // move end back to exclusive model
            end += 1;
            // stop if empty or invalid range after movement
            if (start >= end) {
                return;
            }
            for (var i = start; i < end; ++i) {
                if (situationID2Idx2Selected[situationID][i]) {
                    return;
                }
            }
            console.log(start, end)
            start_end_pairs = []
            start_end_pairs.push([start, end])
            let selection_text = "<b>Selected span:</b> <a class=\"selection_a\">";
            start = start_end_pairs[0][0]
            end = start_end_pairs[0][1]
            let select_text = $('#' + situationID).text().substring(start, end)
            selection_text += select_text + "</a>"
            
            document.getElementById("selection_text").innerHTML = selection_text
            $('#quality-selection').css({
                'display': "inline-block",
                'left': pageX - 45,
                'top' : pageY + 20
            }).fadeIn(200, function() {
                disable_everything()
                $('#selection_text').show()
                $('#missing_info_error_types').hide()
                $('#span_error_types').show()
            });
            annotate_select_span(C[situationID], situation_text[situationID], [start, end], situationID)
            getAnnotations();
        });

        $(".add-missing-info-btn").on('click', function (e) {
            situationID = "missing-info-" + this.id[this.id.length-1]
            $('#quality-selection').css({
                'display': "inline-block",
                'left': e.pageX - 45,
                'top' : e.pageY + 20
            }).fadeIn(200, function() {
                disable_everything()
                $('#missing_info_error_types').show()
                $('#span_error_types').hide()
                $('#selection_text').hide()
            });
            getAnnotations();
        });

        $(".correction-reset-btn").click(function () {
            $('#correction_inline-' + String(cur_task_idx)).val(situation_text["situation-" + String(cur_task_idx)])
            getAnnotations();
        });

        $("#antecedent-reset-btn").click(function () {
            $('#redundant-explanation').val(situation_text[situationID].substring(0, start_end_pairs[0][0]));
            $("#button_div").removeClass("disable");
            if ($('#redundant-explanation').val() === "") {
                document.getElementById('error_selection_warning_div').innerHTML = MISSING_OR_INVALID_ANTECEDANT
                $("#button_div").addClass("disable");
            } else {
                document.getElementById('error_selection_warning_div').innerHTML = MINIMAL_ANTECEDANT_SPAN_WARNING
            }
        });
        
        $('.correction_inline').keyup(getAnnotations);
        
        function _get_pid_sid_explanation(error_type) {
            sent_ids = $("#" + error_type + "-explanation-sent").val().split(',').map(s_idx => parseInt($.trim(s_idx))).sort()
            return JSON.stringify({"passage_id": parseInt($("#" + error_type + "-explanation-passage").val()), "sentence_id": sent_ids})
        }

        $('#confirm_button').on("click", function(e) {
            // get text input value
            var error_type = $('input[name="error_type"]:checked').val();
            var explanation = ''
            if (error_type === 'Redundant') {
                explanation = $('#redundant-explanation').val();
            } else if (error_type === 'Wrong-Grounding') {
                explanation = _get_pid_sid_explanation("intrinsic")
            } else if (error_type.startsWith("Missing-")) {
                explanation = _get_pid_sid_explanation("missing-info")
            }

            let display = $('#' + situationID + "-display")
            display.attr('id', situationID + '-display')
            display.attr('data-situation-id', situationID)
            if (situationID.startsWith('situation-')) {
                C[situationID].add(new CharacterSelection(error_type, explanation, start_end_pairs, C[situationID].data.length));
            } else {
                C[situationID].add(new CharacterSelection(error_type, explanation, [[0,0]], C[situationID].data.length));
            }

            C[situationID].update();
            $('#quality-selection').fadeOut(1, function() {
               disable_everything()
            });
            start_end_pairs = []
            if (situationID.startsWith('situation-')) {
                annotate(C[situationID], situation_text[situationID], situationID);
            }
            if (C["situation-" + String(cur_task_idx)].data.length + C["missing-info-" + String(cur_task_idx)].data.length > 0) {
                $("#no_badness-" + String(cur_task_idx)).prop('checked', false);
            }
            getAnnotations();
        });
        
        $(document).on('mouseover','.quality-span',function(e){
            var color_class = $(this).attr("data-error-type")
            var quality_id = e.target.id
            var situation_id = $(this).attr("data-situation-id")
            var span_num = $(this).attr("data-num")
            var p_span_id = ".p-span-" + span_num + '-' + situation_id
            $(p_span_id).addClass("bg-"+color_class);
            $(p_span_id).addClass("white");
        });

        $(document).on('mouseout','.quality-span',function(e){
            var color_class = $(this).attr("data-error-type")
            var quality_id = e.target.id
            var situation_id = $(this).attr("data-situation-id")
            var span_num = $(this).attr("data-num")
            var p_span_id = ".p-span-" + span_num + '-' + situation_id
            $(p_span_id).removeClass("bg-"+color_class);
            $(p_span_id).removeClass("white");
        });
        
        function _is_every_pid_sid_valid(pid, sid) {
            if (!($.isNumeric($.trim(pid)) && Math.floor($.trim(pid)) === Number($.trim(pid)) && pid.indexOf('.') < 0)) {
                return false;
            }

            is_sid_non_integer = sid.split(',').map(idx => $.isNumeric($.trim(idx)) && 
                                                           Math.floor($.trim(idx)) === Number($.trim(idx)) && 
                                                           idx.indexOf('.') < 0)
            if (is_sid_non_integer.indexOf(false) > -1) {
                return false;
            }

            if (!parseInt($.trim(pid)) in passageId2nSents[cur_task_idx]) {
                return false;
            }

            is_valid = sid.split(',').map(idx => parseInt($.trim(idx)) >= 0 &&
                                                 parseInt($.trim(idx)) <= passageId2nSents[cur_task_idx][parseInt($.trim(pid))]).indexOf(false) < 0
            return is_valid;
        }

        $(".antecedent_no_able").on('click',function(e){
            $("input[name='error_type']").removeClass("selected")
            if (situationID.startsWith("situation-")) {
                annotate_select_span(C[situationID], situation_text[situationID], start_end_pairs[0], situationID)
            }

            var warning_element = document.getElementById('error_selection_warning_div')

            reset_redundant_grounding_explanation();
            warning_element.innerHTML = ''
            $("#button_div").removeClass("disable");

            

            if (this.id === 'error-2') {
                $('#redundant_explanation_div').show();
                $('#redundant-explanation').val(situation_text[situationID].substring(0, start_end_pairs[0][0]));
                $("#button_div").removeClass("disable");
                if ($('#redundant-explanation').val() === "") {
                    warning_element.innerHTML = MISSING_OR_INVALID_ANTECEDANT
                    $("#button_div").addClass("disable");
                } else {
                    warning_element.innerHTML = MINIMAL_ANTECEDANT_SPAN_WARNING
                }
            } else if (this.id === 'error-3') {
                $('#grounding_explanation_div').show();
                warning_element.innerHTML = MISSING_OR_INVALID_PID_SID
                $("#button_div").addClass("disable");
            } else if (this.id === 'error-8') {
                warning_element.innerHTML = CHECK_EVERY_PASSAGE_WARNING
            } else if (['error-5', 'error-6', 'error-7'].indexOf(this.id) > -1) {
                $('#missing_info_explanation_div').show();
                warning_element.innerHTML = MISSING_OR_INVALID_PID_SID
                $("#button_div").addClass("disable");
            } 
        })

        $("#redundant-explanation").on('change keyup paste', function() {
            var warning_element = document.getElementById('error_selection_warning_div')
            warning_element.innerHTML = ''
            $("#button_div").removeClass("disable");

            var antecedant_string = situation_text[situationID].substring(0, start_end_pairs[0][0])
            if (this.value === "" || antecedant_string.indexOf($.trim(this.value)) < 0) {
                warning_element.innerHTML = MISSING_OR_INVALID_ANTECEDANT
                $("#button_div").addClass("disable");
            } else if ($.trim(this.value) == $.trim(antecedant_string)) {
                warning_element.innerHTML = MINIMAL_ANTECEDANT_SPAN_WARNING
            }
            // getAnnotations();
        });
        $(".no_badness").change(function () {
            if ($("#" + this.id).prop('checked')) {
                C["situation-" + String(cur_task_idx)] = new Characters("situation-" + String(cur_task_idx), 0);
                C["missing-info-" + String(cur_task_idx)] = new Characters("missing-info-" + String(cur_task_idx), 0);
                annotate(C["situation-" + String(cur_task_idx)], situation_text["situation-" + String(cur_task_idx)], "situation-" + String(cur_task_idx))
                C["situation-" + String(cur_task_idx)].update();
                C["missing-info-" + String(cur_task_idx)].update();
                $('#correction_inline-' + String(cur_task_idx)).val(situation_text["situation-" + String(cur_task_idx)])
            }
            getAnnotations();
        });

        $("#missing-info-explanation-passage,#missing-info-explanation-sent,#intrinsic-explanation-passage,#intrinsic-explanation-sent").on('change keyup paste', function() {
            var error_type = this.id.replace('-explanation', '').replace('-passage', '').replace('-sent', '')
            var pid = $("#" + error_type + "-explanation-passage").val()
            var sid = $("#" + error_type + "-explanation-sent").val()
            $("#button_div").addClass("disable");

            var warning_element = document.getElementById('error_selection_warning_div')

            if (_is_every_pid_sid_valid(pid, sid)) {
                warning_element.innerHTML = ''
                $("#button_div").removeClass("disable");
            } else {
                warning_element.innerHTML = MISSING_OR_INVALID_PID_SID
            }
            // getAnnotations();
        });

        $(".rank-select").change(function () {
            getAnnotations();
        })
    
        $(document).on("keypress", function(e){
            if (e.key === "Enter") {
              e.preventDefault();
            }
        });
    
        $( function() {
            $( "#quality-selection" ).draggable();
        } );

        getAnnotations();
        $('#feedback').keyup(getAnnotations);

        $('#uw-checkbox').prop('checked', false);
        $('#uw-checkbox').change(function(){
          if (this.checked) {
            alert(`If you are an employee of the UW, family member of a UW employee, or UW student involved in this particular research, you cannot participate in this job. Please return your HIT.`)
            $('#submit').prop('disabled', this.checked);
          }
          getAnnotations();
        })
    
        $('#container').append(`
            <button type="submit" disabled id="actual-submit" style="display:none""></button>`);
      
        // Submitting the annotation
        $("#submit").click(function () {
            getAnnotations();
            // for AWS
            $('#actual-submit').prop('disabled', false);
            $('#actual-submit').click();
        });
    });
}


function hasConflict(ab, bc, ac) {
    if (ab == 2) ab = -1;
    if (bc == 2) bc = -1;
    if (ac == 2) ac = -1;

    if (ab > 0 && bc > 0 && ac <= 0) return true;
    if (ab > 0 && bc == 0 && ac < 0) return true;
    if (ab == 0 && bc > 0 && ac < 0) return true;
    if (ab == 0 && bc < 0 && ac > 0) return true;
    if (ab < 0 && bc == 0 && ac > 0) return true;
    if (ab < 0 && bc < 0 && ac >= 0) return true;
    return false;
}

function getConflictErrorMsg(idx1, idx2, idx3) {
    return 'Error: Check your comparisons among prediction #' + String(idx1) + ', #' + String(idx2) + ' and #' + String(idx3) + ' to avoid contradictions. It usually happens when you indicate something like "A better than B" and "B better than C" (which indicates "A better than C") while you also indicate "C better than/equal to A".'
}

function checkRanks(ranks) {
    // do not count the first prediction
    if (prediction_btn_checked[cur_task_idx].indexOf(false, 1) > -1) {
        return UNREAD_PREDICTIONS
    }

    var ab = ranks[0]
    var ac = ranks[1]
    var ad = ranks[2]
    var bc = ranks[3]
    var bd = ranks[4]
    var cd = ranks[5]
    if (hasConflict(ab, bc, ac)) return getConflictErrorMsg(1, 2, 3);
    if (hasConflict(ab, bd, ad)) return getConflictErrorMsg(1, 2, 4);
    if (hasConflict(ac, cd, ad)) return getConflictErrorMsg(1, 3, 4);
    if (hasConflict(bc, cd, bd)) return getConflictErrorMsg(2, 3, 4);
    /*
    is_adjacent_rank = ranks.sort().map(function(currentValue, index, arr) {
        if (index === 0) {
            return true
        } else {
            return (currentValue - arr[index-1]) <= 1;
        }
    })
    if (is_adjacent_rank.indexOf(false) > -1 || ranks.indexOf(1) < 0) {
        return INVALID_RANK_SEQUENCE
    }*/
    if (ranks.filter(v => v !== 0).length == 0) {
        return ALL_SAME_WARNING + ALLOW_TO_SUBMIT
    }
    return ALLOW_TO_SUBMIT
}

function getAnnotations() {

    var isUW = $('#uw-checkbox').prop('checked');

    if (isUW) {
      $('#submit').prop('disabled', true);
      $('#warning_div').html("If you are an employee of the UW, family member of a UW employee, or UW student involved in this particular research, you cannot participate in this job. Please return your HIT.");
      return;
    }

    var pred1_rank = parseInt($('input[name=rank-1-' + String(cur_task_idx) + ']:checked').val());
    var pred2_rank = parseInt($('input[name=rank-2-' + String(cur_task_idx) + ']:checked').val());
    var pred3_rank = parseInt($('input[name=rank-3-' + String(cur_task_idx) + ']:checked').val());

    var pred4_rank = parseInt($('input[name=rank-4-' + String(cur_task_idx) + ']:checked').val());
    var pred5_rank = parseInt($('input[name=rank-5-' + String(cur_task_idx) + ']:checked').val());
    var pred6_rank = parseInt($('input[name=rank-6-' + String(cur_task_idx) + ']:checked').val());
    var preds_ranks = [pred1_rank, 
                       pred2_rank, 
                       pred3_rank,
                       pred4_rank,
                       pred5_rank,
                       pred6_rank,] 

    preds_ranks_list[cur_task_idx] = preds_ranks

    var corrected_pred = $('#correction_inline-' + String(cur_task_idx)).val();
    corrected_pred_list[cur_task_idx] = corrected_pred

    
    var no_correction = $.trim(String(corrected_pred)) === $.trim(String(situation_text["situation-" + String(cur_task_idx)]))
    
    var error_msg = ''
    var warning_msg = ''
    if (!isComparison) {
        if ($("#no_badness-" + String(cur_task_idx)).is(':checked')) {
            if (!no_correction) {
                error_msg = UNEXPECTED_CORRECTION
            }
        } else {
            if (no_correction) {
                error_msg = NO_EXPECTED_CORRECTION
            } else if (C["situation-" + String(cur_task_idx)].data.length + C["missing-info-" + String(cur_task_idx)].data.length === 0) {
                error_msg = NO_EXPECTED_ERROR_LABEL
            }
        }
    } else {
        // deepcopy, for not affected by the array.sort function
        rankError = checkRanks(JSON.parse(JSON.stringify(preds_ranks)))
        if (rankError.indexOf(ALLOW_TO_SUBMIT) > -1) {
            warning_msg = rankError
        } else {
            error_msg = rankError
        }
    }

    var is_valid = (error_msg === '')

    if (!is_valid) {
        document.getElementById('warning_div').innerHTML = error_msg
    } else {
        if (!isComparison) {
            warning_msg = ALLOW_TO_SUBMIT
        }
        document.getElementById('warning_div').innerHTML = warning_msg
    }

    $('#submit').prop('disabled', !is_valid)

    if (!isComparison) {
        $('#response').val(JSON.stringify({'annotations': [{"error": annotation, 
                                                            "corrected prediction": corrected_pred_list}]}));
    } else {
        $('#response').val(JSON.stringify({'annotations': [{"ranks": preds_ranks_list}]}));
    }
    // alert($('#response').val())
}