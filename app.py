from flask import Flask, render_template, request
import sympy as sp
import re  # Import regex for formatting function input

app = Flask(__name__)

# Define allowed functions
allowed_functions = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "exp": sp.exp
}

def process_function(func_str):
    """Convert user-friendly math notation to Python/SymPy notation."""
    func_str = func_str.replace("^", "**")  # Convert ^ to ** (e.g., x^2 → x**2)

    # Ensure explicit multiplication (e.g., 2x → 2*x)
    func_str = re.sub(r"(\d)([a-zA-Z\(])", r"\1*\2", func_str)

    # Ensure function names (e.g., cos, sin) are prefixed with 'sp.'
    for func_name in allowed_functions:
        func_str = re.sub(rf'\b{func_name}\b', f"sp.{func_name}", func_str)

    return func_str

def process_function_for_display(func_str):
    """Formats function for display (e.g., converts x**2 → x², x**3 → x³)."""
    func_str = func_str.replace("**3", "³").replace("**2", "²").replace("**4", "⁴")
    func_str = func_str.replace("**5", "⁵").replace("**6", "⁶").replace("**7", "⁷")
    func_str = func_str.replace("**8", "⁸").replace("**9", "⁹").replace("**10", "¹⁰")
    func_str = func_str.replace("*", "")  # Remove explicit multiplication for display
    return func_str

def newton_raphson(func_str, x0, rel_error):
    """Performs the Newton-Raphson method for root finding."""
    x = sp.Symbol('x')

    func_str = process_function(func_str)  # Convert user input to Python format

    try:
        func = eval(func_str, {"sp": sp, "x": x})  # Convert to SymPy expression
    except Exception as e:
        return None, f"Error parsing function: {e}"

    deriv_func = sp.diff(func, x)  # Compute derivative

    xn = sp.N(x0)  # Use full precision
    prev_xn = None  # Stores previous x_n for error calculation
    iterations = []
    iteration_count = 0
    max_iterations = 100  # Limit the number of iterations

    while iteration_count < max_iterations:
        try:
            # Compute f(xₙ) and f'(xₙ) using full precision
            f_xn = func.subs(x, xn).evalf()
            f_prime_xn = deriv_func.subs(x, xn).evalf()

            if f_prime_xn == 0:
                return None, "Derivative is zero; Newton-Raphson cannot continue."

            xn_next = xn - (f_xn / f_prime_xn)  # Newton-Raphson formula

            # Compute εₐ (%) using (xₙ(current) - xₙ(previous)) / xₙ(current) * 100
            if prev_xn is not None:
                ea = abs((xn_next - prev_xn) / xn_next) * 100  # Correct formula
                ea = round(ea, 2)  # Format to 2 decimal places
            else:
                ea = "N/A"  # No error for first iteration

            # Store iteration details
            iterations.append((
                iteration_count,
                "{:.4f}".format(float(xn)),  # 4 decimal places for xₙ
                "{:.4f}".format(float(f_xn)),  # 4 decimal places for f(xₙ)
                "{:.4f}".format(float(f_prime_xn)),  # 4 decimal places for f'(xₙ)
                f"{ea}%" if ea != "N/A" else "N/A"  # 2 decimal places for εₐ (%)
            ))

            # Stopping condition
            if prev_xn is not None and ea != "N/A" and ea < rel_error:
                break

            prev_xn = xn
            xn = xn_next
            iteration_count += 1

        except Exception as e:
            return None, f"Error during calculation: {e}"

    if iteration_count >= max_iterations:
        return None, "Maximum iterations reached; solution may not have converged."

    return "{:.4f}".format(float(xn)), iterations  # Return final root rounded to 4 decimal places

@app.route("/", methods=["GET", "POST"])
def index():
    """Handles the Flask app's main page and form submission."""
    root = None
    iterations = []
    error_message = None
    function = initial_guess = rel_error = ""
    formatted_function = ""
    derivative_function = ""

    if request.method == "POST":
        function = request.form["function"]
        initial_guess = request.form["initial_guess"]
        rel_error = request.form["rel_error"]

        try:
            initial_guess = float(initial_guess)
            rel_error = float(rel_error)  # Keep as percentage

            # Format function for display
            formatted_function = process_function_for_display(function)

            # Compute and format f'(x)
            derivative_function = process_function_for_display(
                str(sp.diff(eval(process_function(function), {"sp": sp, "x": sp.Symbol('x')})))
            )

            root, result = newton_raphson(function, initial_guess, rel_error)

            if isinstance(result, str):  # If error message
                error_message = result
            else:
                iterations = result  # If valid iterations

        except ValueError:
            error_message = "Invalid number input. Please enter valid values."

    return render_template(
        "index.html",
        root=root,
        iterations=iterations,
        error_message=error_message,
        function=formatted_function,
        initial_guess=initial_guess,
        rel_error=rel_error,
        derivative=derivative_function  # Pass f'(x) to HTML
    )

if __name__ == "__main__":
    app.run(debug=True)
