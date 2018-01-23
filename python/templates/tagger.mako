<%include file="components/header.mako"/>
<div class="fluid-container">
<%
    page_no = data["page_no"]
    url_params = data["url_params"]
    url_params = "&{}".format(url_params) if url_params else ""
%>
<div class="row m-2">
    <div class="col">
    % if page_no > 0:
    <a href="/tag?page_no=${page_no - 1}${url_params}">prev</a>
    % endif
    <a href="/tag?page_no=${page_no + 1}${url_params}">next</a>
    </div>
    <form class="form-inline float-right">
        <input type="text" name="email" placeholder="Email">
        <input type="text" name="profile" placeholder="Profile">
        <button type="submit" class="btn btn-primary">Filter</button>
        <button type="submit" class="btn">Clear</button>
    </form> 
</div>
% for i in range(0, len(data["worksheets"]), 6):
    <%
        worksheets = data["worksheets"][i:i + 6]
    %>
    <div class="row m-4">
    % for wkst in worksheets:
        <div class="col-md-4 col-lg-2"><a href="/editor/${wkst.id}"><img class="img-thumbnail" src="/assets/${wkst.image_path}" /></a></div>
    % endfor
    </div>
% endfor
<a href="/tag?page_no=${page_no - 1}${url_params}">prev</a> <a href="/tag?page_no=${page_no + 1}${url_params}">next</a>
</div>
<%include file="components/footer.mako"/>