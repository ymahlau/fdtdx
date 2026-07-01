document.addEventListener("DOMContentLoaded", function() {
    // Find all the code snippets inside the right sidebar Table of Contents
    const tocNodes = document.querySelectorAll(".page-toc .nav-link code");
    
    tocNodes.forEach(function(node) {
        // Sphinx usually wraps the text in a span.pre, but we check both just in case
        const textTarget = node.querySelector('.pre') || node;
        const text = textTarget.textContent;
        
        // If the text contains a dot (e.g., "ExtrudedPolygon.axis"), keep only the last part ("axis")
        if (text.includes(".")) {
            const parts = text.split(".");
            textTarget.textContent = parts[parts.length - 1];
        }
    });
});