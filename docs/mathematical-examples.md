# Math Examples

Some examples using `cds.modeling`.

## Projectile motion

```python
from cds.modeling import Variable, Constant, BinaryOp, UnaryFunc
from cds.modeling import differentiate_numerically
import math

t = Variable("t")
v0 = Constant(20.0)
theta = Constant(math.pi / 4)
g = Constant(9.81)

# x(t) = v0 * cos(theta) * t
x_pos = BinaryOp("*", BinaryOp("*", v0, UnaryFunc("cos", theta)), t)

# y(t) = v0 * sin(theta) * t - 0.5*g*t^2
y_pos = BinaryOp("-",
    BinaryOp("*", BinaryOp("*", v0, UnaryFunc("sin", theta)), t),
    BinaryOp("*", Constant(0.5), BinaryOp("*", g, BinaryOp("^", t, Constant(2)))),
)

print(f"x(1) = {x_pos.evaluate({'t': 1.0}):.2f} m")
print(f"y(1) = {y_pos.evaluate({'t': 1.0}):.2f} m")
```

## Work done by a force

```python
from cds.modeling import Variable, Constant, BinaryOp, integrate_numerically

x = Variable("x")
# F(x) = 3x^2 + 2x
force = BinaryOp("+",
    BinaryOp("*", Constant(3), BinaryOp("^", x, Constant(2))),
    BinaryOp("*", Constant(2), x),
)

work = integrate_numerically(force, "x", 0, 4, {})
print(f"Work = {work:.2f} J")  # ~80 J
```

## RC circuit charging

```python
from cds.modeling import Variable, Constant, BinaryOp, UnaryFunc

t = Variable("t")
V0 = Constant(12.0)
RC = Constant(2.0)

# V(t) = V0 * (1 - e^(-t/RC))
v_cap = BinaryOp("*", V0,
    BinaryOp("-", Constant(1),
        UnaryFunc("exp", BinaryOp("/", BinaryOp("*", Constant(-1), t), RC))
    )
)

for time in [0.5, 1.0, 2.0, 5.0]:
    print(f"V({time}s) = {v_cap.evaluate({'t': time}):.2f} V")
```
