# In-place transformations

- If a function modifies its input in place and returns a reference to it, this is allowed only to support chaining.
- If chaining is not required, in-place operations return `None`.
- Non-in-place transformations return a reference to a new object.
- If a function accepts an `inplace` parameter, it returns either the object of the original type or `None` depending on that value.

# Copying

- As a general rule, pass `copy=False` to `.astype()`, provided the object is not a view and downstream operations contain no in-place transformations.
