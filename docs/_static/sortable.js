/**
 * Lightweight sortable tables for Sphinx documentation.
 * Add :class: sortable to any list-table directive.
 * Click column headers to sort ascending/descending.
 */
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("table.sortable").forEach(function (table) {
    var headers = table.querySelectorAll("thead th");
    headers.forEach(function (th, colIndex) {
      th.style.cursor = "pointer";
      th.setAttribute("title", "Click to sort");
      th.addEventListener("click", function () {
        sortTable(table, colIndex, th);
      });
    });
  });

  function sortTable(table, colIndex, th) {
    var tbody = table.querySelector("tbody");
    var rows = Array.from(tbody.querySelectorAll("tr"));
    var headers = table.querySelectorAll("thead th");

    // Determine current sort direction
    var asc = th.getAttribute("data-sort-dir") !== "asc";
    headers.forEach(function (h) {
      h.removeAttribute("data-sort-dir");
      h.classList.remove("sort-asc", "sort-desc");
    });
    th.setAttribute("data-sort-dir", asc ? "asc" : "desc");
    th.classList.add(asc ? "sort-asc" : "sort-desc");

    rows.sort(function (a, b) {
      var cellA = a.children[colIndex];
      var cellB = b.children[colIndex];
      if (!cellA || !cellB) return 0;

      var textA = cellA.textContent.trim();
      var textB = cellB.textContent.trim();

      // Try numeric comparison (strip non-numeric except . and -)
      var numA = parseFloat(textA.replace(/[^0-9.\-]/g, ""));
      var numB = parseFloat(textB.replace(/[^0-9.\-]/g, ""));

      if (!isNaN(numA) && !isNaN(numB)) {
        return asc ? numA - numB : numB - numA;
      }

      // Fall back to string comparison
      return asc
        ? textA.localeCompare(textB)
        : textB.localeCompare(textA);
    });

    rows.forEach(function (row) {
      tbody.appendChild(row);
    });
  }
});
