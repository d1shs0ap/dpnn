from math import log, sqrt

def calculate_adv_err(x_0, y_0, a, b, c, d):
    def calculate_single_int(y):
        return -(2*(x_0-b)^3*log(abs(sqrt((y-y_0)^2+(x_0-b)^2)+y-y_0))-y_0^3*log(abs(sqrt((y-y_0)^2+(x_0-b)^2)+x_0-b))+y_0^3*log(abs(sqrt((y-y_0)^2+(x_0-b)^2)-x_0+b))-2*(x_0-a)^3*log(abs(sqrt((y-y_0)^2+(x_0-a)^2)+y-y_0))+y_0^3*log(abs(sqrt((y-y_0)^2+(x_0-a)^2)+x_0-a))-y_0^3*log(abs(sqrt((y-y_0)^2+(x_0-a)^2)-x_0+a))+2*y*(y^2-3*y_0*y+3*y_0^2)*asinh((x_0-b)/abs(y-y_0))-2*y*(y^2-3*y_0*y+3*y_0^2)*asinh((x_0-a)/abs(y-y_0))+4*(x_0-b)*(y-y_0)*sqrt((y-y_0)^2+(x_0-b)^2)-4*(x_0-a)*(y-y_0)*sqrt((y-y_0)^2+(x_0-a)^2))/12

    return calculate_single_int(d) - calculate_single_int(c)

def find_bounding_rectangles(node, domain):

    def dfs(node, left, right, bottom, top):
        if node.is_leaf:
            bounding_rectangles.append([left, right, bottom, top])
            return
        
        if node.axis == 0:
            bound = node.data[node.axis]
            dfs(node.left, left, bound, bottom, top)
            dfs(node.right, bound, right, bottom, top)

        else:
            bound = node.data[node.axis]
            dfs(node.left, left, right, bottom, bound)
            dfs(node.right, left, right, bound, top)

    bounding_rectangles = []

    left, right = domain[0]
    bottom, top = domain[1]
    dfs(node, left, right, bottom, top)

    return bounding_rectangles