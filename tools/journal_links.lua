-- Point relative Markdown links at the rendered HTML pages:
-- report.md and index.md become index.html, any other foo.md becomes foo.html.
-- Query strings and fragments are kept; absolute URLs are left alone.

function Link(link)
  local target = link.target
  if target:match("^%a[%w+.-]*:") or target:match("^/") or target:match("^#") then
    return nil
  end
  local path, rest = target:match("^([^?#]*)(.*)$")
  local dir, name = path:match("^(.-)([^/]*)$")
  if name == "report.md" or name == "index.md" then
    link.target = dir .. "index.html" .. rest
  elseif name:match("%.md$") then
    link.target = dir .. name:gsub("%.md$", ".html") .. rest
  else
    return nil
  end
  return link
end
