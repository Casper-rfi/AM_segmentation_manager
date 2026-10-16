# Contributing

## Licence

The package is under the [Apache License 2.0](LICENSE). A contribution
submitted for inclusion is licensed under the same terms (section 5), so no
separate agreement is needed.

## Running the tests

From the repository root, in the environment the gui and ml extras build:

    uv sync --extra gui --extra ml
    uv run python scripts/preflight.py --full

That runs the lint, the type check, the suite with no extra installed (in an
environment of its own, `.venv-core`) and the whole suite; every one should
pass. To have the first three run before every push, which takes under a
minute:

    git config core.hooksPath .githooks

CI then covers Linux, Windows and macOS, the suite without `ml`, and the tests
needing the `train` extra, which skip here.
[`docs/decisions.md`](docs/decisions.md)
§ Environment says what running the suite anywhere else looks like, which is
worth reading before believing a failure.

## How the package is laid out

The core is four layers, and higher layers depend on lower ones, never the
reverse:

    layout.py      what things are called
    discovery.py   where they are
    scanner.py     what they mean (the model)
    operations/    what you can do to them

`io/` sits below all four. `layout.py` is the single source of truth for
filesystem conventions: nothing else embeds a directory name, so supporting a
new layout is a change there rather than one spread across every module that
touches files.

`training/` sits above `operations/` and beside the four: a training run lives
outside the collection, so it is read from its directory rather than the
ProjectIndex. Nothing in `operations/` imports it.

The rest of the package root supports that spine, and each module's docstring
says how:

| Kind | Modules |
|---|---|
| The spine, and what it carries between the layers | `layout`, `discovery`, `scanner`, `models`, `state`, `history`, `families`, `config`, `metadata`, `image_metadata`, `tasks` |
| Readers of an index, above `operations/` | `aggregate`, `filtering`, `reporting`, `survey`, `plots`, `overview` |
| Support for any layer | `display`, `frames`, `parallel`, `progress`, `colors` |
| Surfaces that start or describe the application | `launcher`, `installation`, `identity` |

No module of the spine imports `operations/`, `training/`, `gui/`, `ml/` or
`cli/` (`test_a_layer_does_not_import_one_above_it`).

Nothing imports `gui/`, so whatever the window can do the command line can do,
and the core installs without Qt. Qt appears only under `gui/widgets/`;
everything the window knows lives in plain Python controllers that test without
a display. Nothing outside `ml/` imports it at module level, the main
environment having no torch.

## Where things are written down

| Document | Holds |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | the working agreement and the package philosophy |
| [`docs/00_roadmap.md`](docs/00_roadmap.md) | what is built, and what to pick up next |
| [`docs/decisions.md`](docs/decisions.md) | why the software is the way it is; read before changing `operations/` |
| [`docs/display.md`](docs/display.md) | the same for the window, and the Qt traps behind it; read before changing `gui/` |
| [`docs/04_future_work.md`](docs/04_future_work.md) | what is not built, and what to answer before designing it |
| [`guides/`](guides/user_guide.md) | how to use it: the user guide, background, the command line, troubleshooting |

## The user guide's generated parts

The guide's tables of metadata fields and commands are generated from the code,
and its pictures are taken of the window, cropped and numbered as
`scripts/guide_screenshots.py` lists them. The keys are not tabled: F1 lists
them in the window.

    uv run python scripts/guide_tables.py
    uv run python scripts/guide_screenshots.py <collection> --training-directory <runs>

`tests/test_guides.py` fails when a table has fallen behind the code, a link
points nowhere, or a picture is shown on no page.
