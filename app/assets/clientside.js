// Use the correct namespace structure that Dash expects
if (!window.dash_clientside) {
    window.dash_clientside = {};
}

// This is where Dash looks for your callback functions
window.dash_clientside.clientside = {
    updateWindowDimensions: function(n_intervals) {
        try {
            // Get window dimensions
            let width = document.documentElement.clientWidth || window.innerWidth;
            let height = document.documentElement.clientHeight || window.innerHeight;
            
            // Make sure we have valid dimensions
            width = width || 1200;  // Default width if somehow undefined
            height = height || 800; // Default height if somehow undefined
            
            // Store dimensions to check for changes
            if (!window._lastDims) {
                window._lastDims = { width, height };
                console.log(`Initial window dimensions: ${width}x${height}`);
                return window._lastDims;
            }
            
            // Always return the latest dimensions on first interval
            if (n_intervals === 0) {
                console.log(`First interval window dimensions: ${width}x${height}`);
                window._lastDims = { width, height };
                return window._lastDims;
            }
            
            // If dimensions haven't changed, prevent update
            if (window._lastDims.width === width && window._lastDims.height === height) {
                return window.dash_clientside.no_update;
            }
            
            // Update stored dimensions and return
            window._lastDims = { width, height };
            console.log(`Window dimensions updated: ${width}x${height}`);
            return window._lastDims;
        } catch (error) {
            console.error("Error in updateWindowDimensions:", error);
            return { width: 1200, height: 800 }; // Fallback dimensions on error
        }
    }
};

// Make sure window dimensions are set immediately on page load
document.addEventListener('DOMContentLoaded', function() {
    window._lastDims = { 
        width: document.documentElement.clientWidth || window.innerWidth || 1200, 
        height: document.documentElement.clientHeight || window.innerHeight || 800 
    };
    console.log(`DOMContentLoaded dimensions: ${window._lastDims.width}x${window._lastDims.height}`);
});

// The captureIsoformPlot function has been removed as we're now storing the plot directly in the callback