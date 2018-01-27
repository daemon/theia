<%include file="components/header.mako"/>
<div class="fluid-container">
<%
    import json
    worksheet = data["worksheet"]
    image = worksheet.image_path
    label_json = json.dumps([l.dict() for l in data["labels"]])
    rect_json = json.dumps([box.dict() for box in data["boxes"]])
    colors = ["#ff0000", "#ffaa00", "#ff00ff", "#00aaff", "#00ff00", "#0000ff"]
%>
<div class="row">
    <canvas id="editor" class="img-thumbnail" width=600 height=800></canvas>
    <div class="col-2">
    % for label in data["labels"]:
        <button class="btn btn-primary col-12 mt-1" style="border:none; background-color: ${colors[label.id % len(colors)]} !important;"
        onclick="setLabel(${label.id});">${label.name}</button>
    % endfor
    <button class="btn btn-primary col-12 mt-3" onclick="submit();">Save</button>
    </div>
</div>
</div>
<script>
var canvas = document.getElementById("editor"),
    ctx = canvas.getContext("2d"),
    img = new Image;
var finalCanvas = document.createElement("canvas"),
    finalCtx = finalCanvas.getContext("2d");
img.onload = start;
img.src = "/assets/${image}";

var labels = ${label_json};
var existingRects = ${rect_json};
var rects = new Array();
var colors = ["#ff0000", "#ffaa00", "#ff00ff", "#00aaff", "#00ff00", "#0000ff"];
var labelId = labels[0].id;

function submit() {
    $.ajax({
        url: "/create_tag/",
        type: 'POST',
        data: JSON.stringify({id: ${worksheet.id}, rects: rects}),
        contentType: 'application/json',
        dataType: "json",
        success: res => alert("Saved!")
    });
}

function setLabel(labelId) {
    window.labelId = labelId;
}

function drawExistingRects(context) {
    for (var rectData of existingRects) {
        rect = rectData.box_data
        rectLabelId = rectData.label_id
        paintLabeledRect(rect.x, rect.y, rect.w, rect.h, rectLabelId);
    }
}

function start() {
    window.paint = false;
    ctx.drawImage(img, 0, 0, 600, 800);
    finalCanvas.width = 600;
    finalCanvas.height = 800;
    finalCtx.drawImage(img, 0, 0);
    drawExistingRects(ctx);
    drawExistingRects(finalCtx);
}

function paintHoverCross(x, y) {
    ctx.strokeStyle = labelColor(labelId)
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(600, y)
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 800)
    ctx.closePath();
    ctx.stroke();

    ctx.strokeStyle = labelColor(labelId + 1);
    ctx.beginPath()
    ctx.moveTo(x - 8, y);
    ctx.lineTo(x + 8, y);
    ctx.moveTo(x, y - 8);
    ctx.lineTo(x, y + 8);
    ctx.stroke()
    ctx.closePath();
}

function paintRect(x, y) {
    ctx.strokeStyle = labelColor(labelId);
    ctx.lineJoin = "round"
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(Math.min(anchorX, x), Math.min(anchorY, y), Math.max(anchorX, x) - Math.min(anchorX, x), Math.max(anchorY, y) - Math.min(anchorY, y));
    ctx.stroke();
    ctx.closePath();
}

function labelColor(labelId) {
    return colors[labelId % colors.length]
}

function paintLabeledRect(x, y, w, h, labelId) {
    finalCtx.strokeStyle = labelColor(labelId)
    finalCtx.lineWidth = 2;
    finalCtx.beginPath();
    finalCtx.rect(x, y, w, h);
    finalCtx.stroke();
    finalCtx.closePath();

    finalCtx.beginPath();
    finalCtx.rect(x - 1, y - 10, w + 2, 10);
    finalCtx.fillStyle = labelColor(labelId);
    finalCtx.fill();
    finalCtx.font = "bold 9px Arial";
    finalCtx.fillStyle = "white";
    finalCtx.fillText(labels[labelId - 1].name, x, y)
    finalCtx.closePath();
    ctx.drawImage(finalCanvas, 0, 0);
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
      x: evt.clientX - rect.left - 5,
      y: evt.clientY
    };
}

$("#editor").mousedown(function(e) {
    if (paint) {
        var mousePos = getMousePos(canvas, e);
        x = mousePos.x;
        y = mousePos.y;
        var x1 = Math.min(anchorX, x);
        var y1 = Math.min(anchorY, y);
        var w = Math.max(anchorX, x) - x1;
        var h = Math.max(anchorY, y) - y1;
        paintLabeledRect(x1, y1, w, h, labelId);
        rects.push({label_id: labelId, x: x1, y: y1, w: w, h: h})
    } else {
        var mousePos = getMousePos(canvas, e);
        window.anchorX = mousePos.x;
        window.anchorY = mousePos.y;
        paintRect(anchorX, anchorY);
    }
    paint = !paint;
});

$("#editor").mousemove(function(e) {
    var mousePos = getMousePos(canvas, e);
    x = mousePos.x;
    y = mousePos.y;
    ctx.drawImage(finalCanvas, 0, 0, 600, 800);
    paintHoverCross(x, y);
    if (paint)
        paintRect(x, y);
})

</script>
<%include file="components/footer.mako"/>