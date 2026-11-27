# Native ABI adapter for the installed Modular/MAX normalization library.
from layout import Coord, TileTensor, row_major
from nn.normalization import rms_norm_cpu
from std.collections import Optional
from max.gpu.host import DeviceContext
from std.os import abort
from std.runtime import initialize_runtime


@export("max_rms_norm")
@no_inline
def max_rms_norm(
    x: Pointer[Float32, MutUntrackedOrigin], x_d0: Int64, x_d1: Int64, x_s0: Int64, x_s1: Int64,
    weight: Pointer[Float32, MutUntrackedOrigin], w_d0: Int64, w_s0: Int64,
    output: Pointer[Float32, MutUntrackedOrigin], y_d0: Int64, y_d1: Int64, y_s0: Int64, y_s1: Int64,
    inverse_features: Float32, epsilon: Float32,
) abi("C"):
    initialize_runtime()
    # MAX 26.7 specializes epsilon at compile time; reject other ABI values.
    if epsilon != Float32(1.0e-6):
        abort("MAX RMSNorm adapter requires epsilon=1e-6")
    var gamma = TileTensor(weight, row_major(Coord(Int(w_d0)))).as_imm()
    var shape = Coord(Int(x_d0), Int(x_d1))
    @always_inline
    def read[width: Int](coordinate: Coord) {var x, var x_s0} -> SIMD[DType.float32, width]:
        return x.unsafe_offset(Int(coordinate[0].value()) * Int(x_s0) + Int(coordinate[1].value())).unsafe_load[width=width]()
    @always_inline
    def write[width: SIMDLength, alignment: Int](coordinate: Coord, value: SIMD[DType.float32, width]) {var output, var y_s0}:
        output.unsafe_offset(Int(coordinate[0].value()) * Int(y_s0) + Int(coordinate[1].value())).unsafe_store(value)
    try:
        var context = DeviceContext(api="cpu")
        rms_norm_cpu[epsilon=Float32(1.0e-6), weight_offset=Float32(0), multiply_before_cast=True](
            read, write, shape, gamma, Optional(context.assert_cpu_target())
        )
    except error:
        abort(String("MAX RMSNorm failed: ", error))
