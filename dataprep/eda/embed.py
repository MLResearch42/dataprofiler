"""Fast conversion of bokeh models into embeddable HTML."""

from typing import Any, Sequence, Tuple, Union

from bokeh.document import Document
from bokeh.embed import components as _bokeh_components
from bokeh.model import Model

__all__ = ["components"]


def components(models: Union[Model, Sequence[Model]]) -> Tuple[str, Any]:
    """Like :func:`bokeh.embed.components`, but linear in the number of models.

    ``bokeh.embed.components`` adds the models to a new document one at a time,
    and the document re-scans every model it holds after each addition. With one
    chart per column that is quadratic in the number of columns (17 of 21 seconds
    for an 81-column table). Adding them all while the document is frozen makes it
    scan once; bokeh then serializes that document as-is.
    """
    many = isinstance(models, (list, tuple))
    objs = list(models) if many else [models]
    if objs and all(isinstance(obj, Model) and obj.document is None for obj in objs):
        if len({id(obj) for obj in objs}) == len(objs):
            doc = Document()
            with doc.models.freeze():
                for obj in objs:
                    doc.add_root(obj)
    return _bokeh_components(objs if many else objs[0])
