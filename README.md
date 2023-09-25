# Pulsar Labs Site

## Overview

A static Hugo site for the [Pulsar Labs](https://pulsar-labs.co.uk) homepage.


## Getting Started

These instructions will help you set up and run your Hugo site locally for development and testing purposes.

### Prerequisites

- [Hugo](https://gohugo.io/getting-started/installing/) ensure Hugo is installed on your system.

### Installation

1. Clone this repository to your local machine.
    ```
    git clone git@github.com:pulsar-labs/site.git
    ```

2. Navigate to project root.
    ```
    cd site

    ```
3. Run [local development server](http://127.0.0.1:1313/) on `http://127.0.0.1:1313`
    ```
    hugo -D
    ```

### Adding Content

Add a new page to the site.

```
hugo new content new-blog-post.md
````
will add the file `new-blog-post.md` to the folder `content/` e.g 
```
content/new-blog-post.md
```

### Themes

Themes can be installed by cloning theme submodules into the `themes/` directory in the project root, and referencing the cloned theme in `hugo.toml`

e.g

```
git submodule add https://github.com/theNewDynamic/gohugo-theme-ananke.git themes/ananke
```
```
echo "theme = 'ananke'" >> hugo.toml
```

### Deployment

1. Build the site.

```
hugo
```


