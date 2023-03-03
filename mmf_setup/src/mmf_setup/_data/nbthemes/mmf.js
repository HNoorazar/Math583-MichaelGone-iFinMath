// MathJaX customization, custom commands etc.
console.log('Updating MathJax configuration');
MathJax.Hub.Config({
  "HTML-CSS": {
      //availableFonts: ["Neo-Euler"], preferredFont: "Neo-Euler",
      //webFont: "Neo-Euler",
      //scale: 85, // Euler is a bit big.
      mtextFontInherit: true,
      matchFontHeight: true,
      scale: 90, // STIX is a bit big.

  },
});
