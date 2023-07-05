(function() {
  var ARROW = "<span class='glyphicon glyphicon-arrow-right'></span>";
  let isComparison = $('#taskKey').attr("value")=='"comparison"';
  var imgURLPrefix = "http://ellenmellon.github.io/FBForASQA-annotation-img";

  $( window ).init(function(){
    loadValidationHTML();
    loadInstructions();
    setWidth();
    // loadAllValidation();
    load(JSON.parse($("#prompt").attr("value"))["question"]);
    // getAnnotations();
    turkSetAssignmentID();
  });

  function loadInstructions() {
    INSTRUCTIONS = getInstructions();  // TODO
    if (isComparison) {
      INSTRUCTIONS = getComparisonInstructions();
    }

    /* For Instructions */
    $('#instruction-header')
      .html('Instructions (Click to expand).')
      .mouseover(function(){
        this.style.textDecoration = "underline";
      })
      .mouseout(function(){
        this.style.textDecoration = "none";
      })
      .click(function(){
        if ($('#instruction-body').css('display')=='block') {
          $('#instruction-body').css('display', 'none');
          $('#instruction-header').html('Instructions (Click to expand).');
        } else {
          $('#instruction-body').css('display', 'block');
          $('#instruction-header').html('Instructions (Click to collapse).');
        }
      });
    $('#instruction-body').css('display', 'none');

    $('.instructions-item').click(function() {
      $('.active').removeClass("active");
      $('#'+this.id).parent().addClass("active");
      $('#instructions').html(INSTRUCTIONS[this.id]);
    });
    $('#instructions').html(INSTRUCTIONS['instructions-overview']);
  }  

  function loadExtraStyleFiles() {
    return `<!-- The ol big boy, jQ 
    <script src="https://code.jquery.com/jquery-3.5.1.js" integrity="sha256-QWo7LDvxbWT2tbbQ97B53yJnYU3WhH/C8ycbRAkjPDc="
        crossorigin="anonymous"></script> -->
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script> 

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-bar-rating/1.2.2/jquery.barrating.min.js"></script>
    
    <!--CSS 
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/3.7.2/animate.min.css"> 
    <link rel="stylesheet" href="styles/select_box.css">
    <link rel="stylesheet" href="styles/selection.css"> -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> 
    <link rel="stylesheet" href="https://unpkg.com/tachyons@4.10.0/css/tachyons.min.css" /> `
  }

  function loadSpanLabelBox() {
    return `<div id="quality-selection" class="quality-selection gradient-border">
    <i class="fa fa-close" id="close-icon"></i>
    <p id="selection_text" class="selection_text"></p>
    <div id="dropdown-button-container">
            <div class="single_part section over-hide z-bigger">
              <div class="quality_selection_container">
                <div id="span_error_types">
                    <p class="b mb2">Select the factual/language error.</p>
                    <div class="row">
                        <div class="column">
                            <input class="checkbox-tools antecedent_no_able" type="radio" name="error_type" id="error-1" value="Irrelevant">
                            <label class="b for-checkbox-tools" for="error-1" id="error-1-label">
                                Irrelevant
                            </label>
                        </div>
                        <div class="column">
                            <input class="checkbox-tools antecedent_no_able" type="radio" name="error_type" id="error-2" value="Redundant">
                            <label class="b for-checkbox-tools" for="error-2" id="error-2-label">
                                Repetitive
                            </label>
                        </div>
                        <div class="column">
                            <input class="checkbox-tools antecedent_no_able" type="radio" name="error_type" id="error-3" value="Wrong-Grounding">
                            <label class="b for-checkbox-tools" for="error-3" id="error-3-label">
                                Inconsistent Fact
                            </label>
                        </div>
                        <div class="column">
                            <input class="checkbox-tools antecedent_no_able" type="radio" name="error_type" id="error-8" value="Unverifiable">
                            <label class="b for-checkbox-tools" for="error-8" id="error-8-label">
                                Unverifiable Fact
                            </label>
                        </div>
                        <div class="column">
                            <input class="checkbox-tools antecedent_no_able" type="radio" name="error_type" id="error-4" value="Incoherent">
                            <label class="b for-checkbox-tools" for="error-4" id="error-4-label">
                                Incoherent
                            </label>
                        </div>
                    </div>
                </div>  

                <div id="missing_info_error_types">
                    <p class="b mb2">Select the missing info error.</p>
                    <div class="row">
                        <div class="column-missing-info column">
                            <input class="checkbox-tools checkbox-tools-missing-info antecedent_no_able" type="radio" name="error_type" id="error-5" value="Missing-Answer">
                            <label class="b for-checkbox-tools" for="error-5" id="error-5-label">
                                Missing Answer
                            </label>
                        </div>
                        <div class="column-missing-info column">
                            <input class="checkbox-tools checkbox-tools-missing-info antecedent_no_able" type="radio" name="error_type" id="error-6" value="Missing-Major-Auxiliary">
                            <label class="b for-checkbox-tools" for="error-6" id="error-6-label">
                                Missing Major Auxiliary Info
                            </label>
                        </div>
                        <div class="column-missing-info column">
                            <input class="checkbox-tools checkbox-tools-missing-info antecedent_no_able" type="radio" name="error_type" id="error-7" value="Missing-Minor-Auxiliary">
                            <label class="b for-checkbox-tools" for="error-7" id="error-7-label">
                                Missing Minor Auxiliary Info
                            </label>
                        </div>
                    </div>
                </div>
              </div>
            </div>

                

        <div class="single_part" id="redundant_explanation_div">
            <p class="b mb2">Your selected span repeats (an earlier span of text): </p>
            <textarea id="redundant-explanation" name="redundant-explanation" placeholder="Please write down the earlier span of text being repeated!" rows="4" style="width: 100%; overflow-y: hidden;"></textarea>
            <p class="mb0"><button type="button" class="btn reset-btn btn-default" id="antecedent-reset-btn"> Reset </button></p>
        </div>

        <div class="single_part" id="grounding_explanation_div">
              <br/>
              <p><label for="intrinsic-explanation-passage" style="vertical-align:top;">with passage # </label>
              <textarea id="intrinsic-explanation-passage" name="intrinsic-explanation-passage" rows="1" style="width: 10%; resize: none; overflow-y: hidden;"></textarea>
              </p>
              <p><label for="intrinsic-explanation-sent" style="vertical-align:top;">sentence # </label>
              <textarea id="intrinsic-explanation-sent" name="intrinsic-explanation-sent" rows="1" style="width: 10%; resize: none; overflow-y: hidden;"></textarea>
              <span style="vertical-align:top;">(in rare cases, list multiple sentences separated with comma if necessary)</span>
              </p>
        </div>

        <div class="single_part" id="missing_info_explanation_div">
            <p class="b mb2"><label for="missing-info-explanation">You find a piece of missing information in: <br></label></p>
            <br>
              <p><label for="missing-info-explanation-passage" style="vertical-align:top;">passage # </label>
              <textarea id="missing-info-explanation-passage" name="intrinsic-explanation-passage" rows="1" style="width: 10%; resize: none; overflow-y: hidden;"></textarea>
              </p>
              <p><label for="missing-info-explanation-sent" style="vertical-align:top;">sentence # </label>
              <textarea id="missing-info-explanation-sent" name="intrinsic-explanation-sent" rows="1" style="width: 10%; resize: none; overflow-y: hidden;"></textarea>
              <span style="vertical-align:top;">(in rare cases, list multiple sentences separated with comma if necessary)</span>
              </p>
        </div>

        <div id="error_selection_warning_div" class="pl4" style="color:Red"></div>
        
        <div class="disable buttons" id="button_div">
          <button id="confirm_button" class="b confirm quality_button" type="button">Confirm</button>
        </div>
    </div>
  </div>`
  }

  function loadPredictionForRank(idx1, idx2, pid, taskId) {
    var namestring = String(pid) + `-` + String(taskId);
    return `<div class="pl4">
              `+ `<button type="button" class="show-hide-btn" id="prediction-` + String(pid) + `-btn-` + String(taskId) + `"> Click to show </button>` +`
              Compare Model Prediction #` + String(idx1) + ` VS #` + String(idx2) + ` -
              <!--label> Rate: </label-->
              <label style="font-weight:normal"><input type="radio" class="rank-select" id="rank-` + namestring + `-1" name="rank-` + namestring + `" value="1"> <span> #` + String(idx1) + ` is better </span></label>
              <label style="font-weight:normal"><input type="radio" class="rank-select" id="rank-` + namestring + `-0" name="rank-` + namestring + `" value="0" checked> <span> equal </span></label>
              <label style="font-weight:normal"><input type="radio" class="rank-select" id="rank-` + namestring + `-2" name="rank-` + namestring + `" value="2"> <span> #` + String(idx2) + ` is better </span></label>

            </div>
            <div id="prediction-` + String(pid) + `-div-` + String(taskId) + `">
              <p id="prediction-` + String(pid) + `-` + String(taskId) + `" style="display:none" class="pl4 pr4 pa1 mt0 mb3 paragraph"> Loading ... </p>
              <p class="f3 pl4 pr4 pa1 mt0 mb0 paragraph"></p>
            </div>`  
  }

  function loadQualTaskHtml(taskId) {
    return `
    <div class="qual-hits hit-` + String(taskId) + `">
    <div class="panel panel-default narrow-panel" id="situation-` + String(taskId) + `-input-div">
      <div id="situation-` + String(taskId) + `-prompt-div" class="panel-heading">
        <p class="pl4 pr4 pt1 mt0 mb0 input_type">Input Question: </p>
        <p id="situation-` + String(taskId) + `-prompt" class="f2 pl4 pr4 pa1 mt0 mb0 prompt">
            Loading ...
        </p>
      </div>

      <div id="situation-` + String(taskId) + `-passage-div" class="panel-body">
      </div>
    
      <div id="situation-` + String(taskId) + `-gold-div" class="panel-footer">
        <p class="pl4 pr4 pt1 mt0 mb0 input_type">Reference Response: </p>
        <p id="situation-` + String(taskId) + `-gold" class="pl4 pr4 pa1 mt0 mb0">
            Loading ...
        </p>
      </div>
    </div>
  
    <div id="situation-` + String(taskId) + `-div" >
      <div id="situation-` + String(taskId) + `-generation-div">
        <p class="pl4 pr4 pt1 mt0 mb0" style="color:#4863A0"><strong><em>STEP 1</em>: Error labeling for the model-predicted response: </strong>
        <button type="button" class="show-hide-btn" id="step1-btn-` + String(taskId) + `"> Click to hide </button>
        <span style="font-size:9pt;color:grey"><em>*** Hint: The mistake labeling prompt window is draggable. ***</em></span>
        </p>
        
        <div id="step1-` + String(taskId) + `">
          <div class="pl4 mt3 mb3">
            <input type="checkbox" class="no_badness" id="no_badness-` + String(taskId) + `" name="no_badness-` + String(taskId) + `">
            <label for="no_badness-` + String(taskId) + `">After <em style="color:Red">CAREFUL</em> checking, I find no error in this predicted response!</label>
          </div>

          <p id="situation-` + String(taskId) + `" class="pl4 pr4 pa1 mt0 mb0 paragraph situation-text"> Loading ... </p>
          <div id="situation-` + String(taskId) + `-display" class="situation-display mb3"></div>
          <p class="pl4 mt3"><button type="button" class="btn btn-default add-missing-info-btn" id="add-missing-info-btn-` + String(taskId) + `">Add Missing Info</button> <span style="font-size:9pt;color:grey"><em>*** Hint: If the error type selection doesn't work, try double clicking. ***</em></span></p>
          <div id="missing-info-` + String(taskId) + `-display" class="situation-display mb3"></div>
        </div>
        
        <br>
  
        <p class="pl4 pr4 pt1 mt0 mb0" style="color:#4863A0"><strong><em>STEP 2</em>: Your correction of the model-predicted response: </strong>
        <button type="button" class="show-hide-btn" id="step2-btn-` + String(taskId) + `"> Click to hide </button></p>
      
        <div id="step2-` + String(taskId) + `">
          <textarea placeholder="You should only leave the box blank if no grounding passage is providing any useful information." id="correction_inline-` + String(taskId) + `" class="correction_inline" rows="4" style="margin-left:20px; margin-right:20px; margin-top:10px; width: 97%; overflow-y: hidden;"></textarea>
          <p class="pl4 mb0"><button type="button" class="btn reset-btn btn-default correction-reset-btn" id="reset-btn-` + String(taskId) + `"> Reset </button></p>
        </div> 
      </div>
      
      <div id="compare-div-` + String(taskId) + `">
        <p class="pl4 pr4 pt1 mt0" style="color:#4863A0"><strong>Your Task:</strong> Make pairwise comparisons on the following 6 model prediction pairs based on their overall quality. <br> </strong>
        <span style="font-size:10pt;color:grey"><em>*** Hint: The 6 pairs are different combinations of 4 model predictions. We index each model prediction and highlight them with different colors for your convenience. ***</em></span>
        
        <!--button type="button" class="show-hide-btn" id="step3-btn-` + String(taskId) + `"> Click to hide </button--></p>
        <div id="step3-` + String(taskId) + `">
        ` + loadPredictionForRank('1', '2', 1, taskId) 
          + loadPredictionForRank('1', '3', 2, taskId) 
          + loadPredictionForRank('1', '4', 3, taskId)
          + loadPredictionForRank('2', '3', 4, taskId)
          + loadPredictionForRank('2', '4', 5, taskId)
          + loadPredictionForRank('3', '4', 6, taskId) 
          + `
        </div>
      </div>
    </div>
    </div>`
  }

  function loadValidationHTML() {
    $('#taskContent').html(
      loadExtraStyleFiles() +
      `<!-- Instruction -->
      <div class="container" id="container" role="main">
        <div class="panel panel-default">
          <div class="panel-heading"><button id="instruction-header" type="button" class="" ></button></div>
          <div class="panel-body" id="instruction-body">
          <nav class="navbar navbar-default">
          <div class="container-fluid">
            <ul class="nav navbar-nav">
                  <li class="active"><a href="#" id="instructions-overview" class="instructions-item">Overview</a></li>
                  ` + ((isComparison)? `` : `
                  <li><a href="#" id="instructions-examples" class="instructions-item">Example</a></li>
                  <li><a href="#" id="instructions-step-by-step" class="instructions-item">FAQ</a></li>`) + 
                  `
                </ul>
              </div>
            </nav>
            <div id="instructions">
              Instructions (TODO)
            </div>
          </div>
        </div> 
      `
      + loadSpanLabelBox() +
       `
       <!-- Example situation to be annotated -->
        ` + loadQualTaskHtml(0) + 
        `
        <br>
        <div id="warning_div" class="pl4" style="color:Red"></div>
        <br>

        <div class="pl4">
              <textarea placeholder="Optional: If you have any feedback for us, please leave it here!" class="" rows="4" id="feedback" name="feedback" style="width: 80%; overflow-y: hidden;"></textarea>
              <br /><br />
              <div id="submit-button-div">
                  <button type="button" class="btn btn-primary" id="submit" disable>Submit!</button>
              </div>
        </div>
      </div>`);
  }

  function getInstructions() {
    return {"instructions-overview": 
    `
    <p>In each task, <strong>you will be given</strong> a question, a set of Wikipedia passages (with their article title provided), a reference response, and a model-predicted response. Each passage is presented as a sequence of sentences (title is indexed as sentence #0). <strong>Your goal is to</strong> 1) label mistakes made in the model prediction and 2) make corrective edits to the model prediction based on your labeled mistakes. <br><br>
    
    <span class="hl"><strong>Important Definitions</strong></span>: <em>An ideal response</em> to a question should provide both <strong><em>answer(s)</em></strong> that directly responds to the question and <strong><em>crucial auxiliary information</em></strong> for better comprehension of the answer(s). We consider auxiliary information as <em>crucial</em> if it is used in the reference response. Additionally, all information in <em>an ideal response</em> should be <strong><em>factually consistent with (i.e., grounded in) the passages</strong></em>. Note that the reference response is written by a human with potentially different grounding passages, and thus, you might find <em><strong>answers</em></strong> that can be found in the passages but are not included in the reference, which are <em><strong>STILL</em></strong> expected to be in the model prediction. On the other hand, answers in the reference that cannot be found in or verifiable by the passages are <em><strong>NOT</em></strong> expected to be in the model prediction. <em><strong>To conclude, all answers</em></strong> are expected in the model prediction <em><strong>IF AND ONLY IF</em></strong> it can be found in the passages. <strong><em>Crucial auxiliary information</em></strong> is expected in the model prediction <em><strong>IF AND ONLY IF</em></strong> it can be found in both the reference response and the passages.
    
    <br><br>
    Here are the detailed annotation steps:</p>

    <br />
    <p>
    <!--iframe width="800" height="450" src="https://www.youtube.com/embed/unknown?cc_load_policy=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe-->
    
    <strong style="color:Blue;"><em>STEP 1 - </em></strong> Read the question and label mistakes made in a model-predicted response. As explained above, leverage the reference, <strong>BUT</strong> rely on the passages. 
      Decide the mistake type and follow detailed instructions as follows. We encourage you to use CTRL+F/CMD+F for navigating the reference and passages using keywords.
      <ul>
        <li>
          <strong>Erroneous Span (i.e., substring)</strong>: Highlight each span in the model prediction that contains one of the following errors. 
          Label each span <em><strong>as short as possible</strong></em> and make sure each labeled span only contain <em><strong>one single</strong></em> information piece. You will be prompted to select the error type and provide an explanation if apply. For each span, label <em><strong>only one</em></strong> error type. If a span contains multiple errors, select the error type based on the order below (e.g., if a span contains "irrelevant" information, ignore any "inconsistent fact" it contains).  
          <ol type='i'>
            <br style="line-height:16px;"/>
            <li>
              <strong>[Irrelevant]</strong>: 
              The span contains "irrelevant" information (e.g., neither an answer nor crucial auxiliary information, defined in the first 2 sentences in "<em>Important Definitions</em>"). To detect "irrelevant" errors, you do not need to consider whether the information is factually consistent with the passages or not.
              <!--Any auxiliary information that is not included in the reference should be marked as "irrelevant" (e.g., not crucial). <em><strong>Note that</strong></em> for every answer in the model prediction, even if it is missing in the reference, do <em><strong>NOT</strong></em> count it as irrelevant.-->
            </li>
            <br style="line-height:8px;"/>
            <li>
              <strong>[Repetitive]</strong>: 
              The span repeats information in its previous text. Provide the previous text being repeated (<em><strong>as short as possible</strong></em>). <strong><em>Note that</strong></em> a "repetitive" span should still be marked even if its previous text being repeated contains an factual or coherence issue (defined below). However, if the previous text is "irrelevant", it should be marked as "irrelevant" too.
            </li>
            <br style="line-height:8px;"/>
            <li>
              <strong>[Inconsistent Fact]</strong>:
              The span is factually inconsistent with the passages. Enter the passage id and sentence id(s) as evidence. <strong><em>Note that</strong></em> if you find multiple evidences in the passages, mark only one of them. The need for multiple passage ids usually indicates that you should separate the error into multiple ones (due to multiple information pieces).
            </li>
            <br style="line-height:8px;"/>
            <li>
              <strong>[Unverifiable Fact]</strong>:
              The span is factually unverifiable (i.e., not mentioned in any passage), after <strong><em>carefully</em></strong> checking all passages. Common sense (e.g., <span class="q">"a bicyle has two wheels"</span>) doesn't need to be verified. However, do not count knowledge only commonly known in a specific region/community as commonsense. This can be subjective, and simply follow your best judgment.
            </li>
            <br style="line-height:8px;"/>
            <li>
              <strong>[Incoherent]</strong>: 
              The span contains major grammar error (ignore minor typos), is uninterpretable, contradicts to common sense, or is not coherent with its context.
            </li>
          </ol>
        </li>
        <br style="line-height:32px;"/>
        <li>
          <strong>Missing Information</strong>: Identify information that is expected but missing in the model prediction. Check "<em>Important Definitions</em>" above to see how to identify such information. Classify <em><strong>each piece</strong></em> of missing information as <strong>[Missing Answer]</strong> or <strong>[Missing Major/Minor Auxiliary Information]</strong>, and enter the passage id and sentence id(s) as evidence. Mark the missing auxiliary information as <em><strong>major</strong></em> if you think the information is indeed helpful for understanding the answer. Otherwise (e.g., a bit off-topic), mark it as <em><strong>minor</strong></em>. Simply follow your best judgment. Follow the same <em><strong>"Note that"</em></strong> rule under <em>"[Inconsistent Fact]"</em> above.
      </ul>
      <br style="line-height:16px;"/>
      <span class="hl"><strong>Important Notes</strong></span>:<br> 
      1. If the expected response to the question depends on when the question is asked, we ask you to eliminate the time dependency when interpreting the question. For example, simply interpret the question <span class="q">"What date was Thanksgiving last year?"</span> as <span class="q">"What date was Thanksgiving?"</span> In that case, the date of Thanksgiving in 2022, 2021, ... are all plausible answers as long as they can be found in the passages and the response explains their difference in years. <br>
      2. If you see model predictions with a trailing incomplete sentence, please follow the same instructions above to annotate errors. <strong>NOTE THAT</strong> we ask you to focus on the <strong>CONTENT</strong> of the incomplete sentence and <strong>DO NOT</strong> label "Incoherence" for its incompleteness. If the incomplete sentence contains no actual information (e.g. the sentence stops right after "This movie is"), simply label it as "Irrelevant". When you make corrections for such an incomplete sentence (in STEP 2), you ask you to make the sentence a complete one. For example, if the sentence goes "The movie is", then simply delete it in STEP 2. If the sentence goes "The movie is released in 1998 and it is" and you think "The movie is released in 1998" is error free, simply correct it as "The movie is released in 1998." 
    </p>
    <br style="line-height:16px;"/>
    <p>
      <strong style="color:Blue;"><em>STEP 2 - </em></strong> Correct the model prediction to address your labeled mistakes in STEP 1. <em><strong>Instead of</strong></em> copying and pasting the reference into the box, make minimal local edits on the original prediction. Make sure the corrected version is fluent and free of typos. In rare cases when you find no useful information in passages for answering the question, you can leave the box as blank if necessary.
    </p>


    <hr />
    <hr />

    <span style="color:Black">See the <strong>Example</strong> tab for step-by-step explanations. Check
    <a href="https://docs.google.com/document/d/1sXrzEZmQiJOCUdX81Ad7tfBMSBykUFOu0q3Mel89UkU/edit?usp=sharing" target="_blank">this table</a>,
    <a href="https://docs.google.com/document/d/1vM5jFSNomRi_1obcY60VMlpAjL476-hRE7Ndx6vMSIw/edit?usp=sharing" target="_blank">this doc</a> as well as the <strong>FAQ</strong> tab for common mistakes from workers and how we review your submission. <br/><br/>
    </span>
    <span><em>If you run into any issue, email me at <a href="mailto: zeqiuwu1@uw.edu"><strong>zeqiuwu1@uw.edu</strong></a> or DM me (<strong>Ellen Wu</strong>) on Slack (Turker Nation). I'll respond within a few minutes (at most 1 hour) during 9am-10pm PT (Sun-Fri).</em></span>
    `, 
    "instructions-examples": `
    Here, we show annotation and step-by-step explanation for an example in STEP 1 & 2 for full understanding of each error type and key instructional points: <br/><br/>
    <strong>[Example Input]:</strong><br/>
    <img src="` + imgURLPrefix + `/ex-input.png" width="1000px" style="border: 1px solid #555;"/> 

    <br/><br/>
    <strong>[STEP 1 Annotation]:</strong><br/>
    <img src="` + imgURLPrefix + `/ex-ann.png" width="1000px" style="border: 1px solid #555;"/>
    <br/>
    <p>
      <span style="color:blue"><strong>Explanation :</strong></span> 
      <ol>
        <li>
          <strong>The second sentence</strong> in the model prediction is neither a direct answer nor crucial auxiliary information (i.e., not found in the reference). Thus, it should be marked as "<strong>Irrelevant</strong>". You can ignore the "$59.37 billion" factual inconsistency error, as you only need to label one error type for each span by following the priority order introduced in the "Overview" tab.
        </li>
        <br/>
        <li>
          The provided passages only mention Jeepers Creepers 2 was released in 2003 without an specific date. Therefore, "<strong>August 29</strong>" in the third sentence is an "<strong>Unverifiable Fact</strong>", even if it appears in the reference response. Remember that you should try to make the labeled span as short as possible. Therefore, do not include "2003" or "on" in the labeled span.
        </li>
        <br/>
        <li>
          "<strong>The film</strong>" in the fourth sentence could be interpreted as "Jeepers Creepers 2" from the previous sentence. However, it is actually mentioning some fact from Passage 1 that refers to "Jeepers Creepers 3". Therefore, as clarified in FAQ #3, the span should be marked as "<strong>Inconsistent Fact</strong>". Note that you should not label the fourth sentence as "Incoherent" (you might see it as contradicting to the previous sentence), as you should always select the error type based on the order introduced in the "Overview" tab.
        </li>
        <br/>
        <li>
          Whether Jeepers Creepers 3 came out "<strong>in the UK on September 4, 2017</strong>" <strong>cannot be verified</strong> from the passages.
        </li>
        <br/>
        <li>
          <strong>The fifth sentence</strong> contains information already expressed in the previous sentence. Therefore, label it as "<strong>Repetitive</strong>". Note that a "repetitive" span should still be marked if its previous text being repeated contains an factual or coherence issue.
        </li>
        <br/>
        <li>
          <strong>The last sentence</strong> is incomplete and appears very "<strong>Incoherent</strong>" to the previous context.
        </li>
        <br/>
        <li>
          Passage 1 and sentence 4 contains a <strong>missing answer</strong> for the fourth film. Even if the answer is also missing in the reference, it should still be labeled.
        </li>
        <br/>
        <li>
          The reference contains auxiliary information "released by United Artists and Metro-Goldwyn-Mayer" that can be found in passage 1 and sentence 1. It does not seem super critical for explaining the answer, thus it can be labeled as "<strong>Missing Minor Auxiliary Info</strong>". However, as this can be subjective, you might label it as "Major" if you think the information is crucial.
        </li>
        <br/>
        <li>
          We understand sometimes the annotation could be subjective. For example, as people usually consider AMC announced release date to be reliable, some may label "in the US" in the fourth sentence as "Unverifiable" or "Inconsistent" as "September 26, 2017" was stated by AMC Theatres without further confirmation. While you should try your best to follow our instructions, such reasonable annotations are all acceptable.
        </li>
      </ol>
    </p> 

    ** The following are filled prompt windows for labeling "Repetitive", "Inconsistent Fact" and "Missing Info" errors, with additional information (i.e., explanation) to enter **
    <br/>
    <img src="` + imgURLPrefix + `/ex-rep.png" width="325px" /> 
    <img src="` + imgURLPrefix + `/ex-inc.png" width="325px" /> 
    <img src="` + imgURLPrefix + `/ex-mans.png" width="325px" /> 

    <br/><br/>
    <strong>[STEP 2 Annotation]:</strong><br/>
    <img src="` + imgURLPrefix + `/ex-correct.png" width="1000px" /> 
    <br/>
    <p>
      <span style="color:blue"><strong>Explanation :</strong></span> In the correction box, make minimal edits to correct all labeled errors in STEP 1. 
    </p>
    `,
    "instructions-step-by-step": `<ol>` +
    `
    <li>
      What if I think there is a misinterpretation of the question in the reference response or the question itself is problematic? 
      ` + ARROW + ` Try to stick with how the reference interprets the question during annotation, unless you believe the reference is a <em>complete</em> misinterpretation (should be very rare), in which case you can come up with your own "reference response" and do the annotation based on that. If you find mistakes (e.g., inconsistent with grounding passages, missing information, etc) in the reference, please follow our rules specified in the overview instruction.
    </li>
    <br/>
    <li>
      I find a span that repeats a previous span, but contains much richer or more specific information, what should I do?
      ` + ARROW + ` You should still label the later span as the repetitive span.
    </li>
    <br/>
    <li>
      I find it sometimes ambiguous to differentiate between "inconsistent fact", "incoherent" or "missing information" errors?
      ` + ARROW + ` Generally speaking, if the model-predicted response misses important contextual information (from the grounding passages) for a span, such that the span would be interpreted differently (e.g., in prediction: <span class="q">The movie was released on August 8, 2020</span>; in passage: <span class="q">The movie was released in the US on August 8, 2020</span>), it should be labeled as "inconsistent fact". If missing such contextual information leads the span to be uninterpretable (e.g., a pronoun without an antecedent), it should be labeled as "incoherent". "Missing information" should be some standalone information that adding/removing it does not affect your interpretation of other information in a response (e.g., in prediction: <span class="q">The movie was released in 2020</span>; in passage: <span class="q">The movie was released on August 8, 2020</span>).
    </li>
    <br/>
    <li>
      What should I do if a given passage seems to be a table or list?
      ` + ARROW + ` Try your best to interpret the information in the table/list. If you find it too hard to interpret, simply ignore that passage. 
    </li>
    <br/>
    <!--li>
      Some span labeling strategy? 
      ` + ARROW + ` TODO
    </li>
    <br/-->
    <li>
      What if I found contradicting facts in passages? 
      ` + ARROW + ` You may want to first check whether such facts are referring to different entities or contexts, which can sometimes be found in passage titles. If they indeed conflict with each other, you can simply choose one of them to be included in the response.
    </li>
    <br/>
    <!--li>
      Can I label a span as irrelevant even if the information it carries is also included in the reference?
      ` + ARROW + ` You should NOT label information that is also included in the reference as "irrelevant", even if you don't think the information is helpful for the question. The only exception is when you believe the reference is a complete misinterpretation of the question, in which case you can come up with your own reference response (see FAQ #3). 
    </li-->
    </ol>`};
  }

  function getComparisonInstructions() {
    return {"instructions-overview": 
    `<p>In each task, <strong>you will be given</strong> a question, a set of Wikipedia passages (with their article title provided), a reference response, and 6 pairs of model-predicted responses. Each passage is presented as a sequence of sentences (title is indexed as sentence #0). <strong>Your goal is to</strong> perform 6 pairwise comparisons of model predictions on their overall quality.<br><br>
    
    <span class="hl"><strong>Important Notes</strong></span>: <strong>The overall quality should be judged based on what type(s) of and how many errors exist in each model prediction.</strong> We assume that you already know how to detect different types of error in a model prediction, from our "erroneous span annotation" task. In this task, you don't need to mark each erroneous span, and we simply ask you to do it in your head for each model prediction for comparison purposes. For example, if two model predictions only differ by one single sentence, you may only need to do close error inspection to that one sentence. Given the severity variation of each individual error, the comparison can sometimes be subjective and you should use your own best judgement. <strong>Try to make a preference over two predictions and use as few ties as possible.</strong> You can always check the detailed instructions, illustration and FAQ of error detection <a href="https://qa.cs.washington.edu:7781/task/annotation/preview#" target="_blank">from "Instructions" in this link</a>.
    

    <hr />
    <hr />

    <span><em>If you run into any issue, email me at <a href="mailto: zeqiuwu1@uw.edu"><strong>zeqiuwu1@uw.edu</strong></a> or DM me (<strong>Ellen Wu</strong>) on Slack. I'll respond within a few minutes (at most 1 hour) during 9am-10pm PT (Sun-Fri).</em></span>
    `};
  }

  function setWidth() {
    $('#container').width($('#taskContent').width()-100);
    $("#feedback").width($('#container').width()/3);
  }

})();


