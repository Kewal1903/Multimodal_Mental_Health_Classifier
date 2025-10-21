/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html"], // This line is the same
  theme: {
    extend: {
      colors: {
        // This section is the same
        'beige-bg': '#F5F1E9',
        'brown-text': '#3D2B1F',
        'brown-accent': '#6B4F4B',
        'brown-light': '#A67B5B',
      },
      // ADD THIS NEW SECTION
      fontFamily: {
        'sans': ['Nunito', 'sans-serif'], // Sets "Nunito" as the default
        'cursive': ['Dancing Script', 'cursive'],
      }
    },
  },
  plugins: [],
}