// Use the correct namespace structure that Dash expects
if (!window.dash_clientside) {
    window.dash_clientside = {};
}

// This is where Dash looks for your callback functions
window.dash_clientside.clientside = {
    updateWindowDimensions: function(n_intervals) {
        // Get window dimensions
        let width = document.documentElement.clientWidth;
        let height = document.documentElement.clientHeight;
        
        // Store dimensions to check for changes
        if (!window._lastDims) {
            window._lastDims = { width, height };
            console.log(`Window dimensions: ${width}x${height}`);
            return window._lastDims;
        }

        // If dimensions haven't changed, prevent update
        if (window._lastDims.width === width && window._lastDims.height === height) {
            return window.dash_clientside.no_update;
        }
        
        // Update stored dimensions and return
        window._lastDims = { width, height };
        console.log(`Window dimensions: ${width}x${height}`);
        return window._lastDims;
    }
};

// The captureIsoformPlot function has been removed as we're now storing the plot directly in the callback