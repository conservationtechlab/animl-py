# AniML

This repository contains the source code for the Conservation Tech Lab's GitHub Pages site, showcasing our open-source wildlife conservation technology projects.

## About

The Conservation Tech Lab develops cutting-edge technology solutions for wildlife conservation and ecological research. Our work spans machine learning for camera trap analysis, edge-AI field devices, bioacoustics tools, and animal tracking systems.

## Technical Details

This site is built with [Jekyll](https://jekyllrb.com/), a static site generator that GitHub Pages supports natively.

### Structure

- `index.md` - Main homepage content with Jekyll front matter
- `_config.yml` - Jekyll site configuration
- `_layouts/default.html` - Page layout template
- `assets/css/style.css` - Site stylesheet
- `Gemfile` - Ruby dependencies

### Local Development

To run the site locally:

1. Install Ruby and Bundler
2. Install dependencies: `bundle install`
3. Run Jekyll: `bundle exec jekyll serve`
4. Visit `http://localhost:4000`

### Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. No manual build step is required.

## License

All projects featured on this site are open-source and available under their respective licenses.