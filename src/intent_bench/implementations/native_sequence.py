"""Time a prepared sequence of C ABI kernels in one native interval."""
import ctypes as ct
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import shutil


def _type(value):
    names = {ct.c_void_p: 'void*', ct.c_bool: 'bool', ct.c_float: 'float', ct.c_double: 'double',
             ct.c_int8: 'int8_t', ct.c_uint8: 'uint8_t', ct.c_int16: 'int16_t', ct.c_uint16: 'uint16_t',
             ct.c_int32: 'int32_t', ct.c_uint32: 'uint32_t', ct.c_int64: 'int64_t', ct.c_uint64: 'uint64_t'}
    if value not in names:
        raise TypeError(f'unsupported native sequence ABI type: {value}')
    return names[value]


class _Call(ct.Structure):
    _fields_ = [('function', ct.c_void_p), ('arguments', ct.POINTER(ct.c_void_p)), ('signature', ct.c_size_t),
                ('fill', ct.c_size_t)]


class _Fill(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('count', ct.c_size_t),
                ('width', ct.c_size_t), ('value', ct.c_ubyte * 8)]


class NativeSequence:
    supports_repeat = True

    def __init__(self, calls):
        bindings = tuple(binding for call in calls for binding in
                         (call.native_calls() if hasattr(call, "native_calls") else (call.native_call(),)))
        fills = []
        signatures, entries, owners = [], [], []
        self.bindings = bindings
        self.arguments = {}
        self._calibrated_repeat = None
        for binding in bindings:
            if not hasattr(binding, "function"):
                if not (isinstance(getattr(binding, "address", None), int)
                        and isinstance(getattr(binding, "count", None), int) and binding.count >= 0
                        and isinstance(getattr(binding, "value", None), bytes)
                        and len(binding.value) in (1, 2, 4, 8)):
                    raise TypeError("unknown native sequence entry")
                fills.append(_Fill(binding.address, binding.count, len(binding.value),
                                   (ct.c_ubyte * 8)(*binding.value)))
                entries.append(_Call(None, None, 0, len(fills)))
                continue
            function = binding.function
            types = binding.argument_types
            result = getattr(binding, 'return_type', None)
            signature = (_type(result) if result else 'void', tuple(_type(t) for t in types))
            if signature not in signatures:
                signatures.append(signature)
            values = binding.arguments
            self.arguments[len(entries)] = values
            pointers = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
            owners.append((binding, values, pointers))
            entries.append(_Call(ct.cast(function, ct.c_void_p), pointers, signatures.index(signature), 0))
        self.owners = owners
        self.calls = (_Call * len(entries))(*entries)
        self.fills = (_Fill * len(fills))(*fills)
        cases = []
        for index, (result, types) in enumerate(signatures):
            arguments = ', '.join(f'*static_cast<{t}*>(call.arguments[{i}])' for i, t in enumerate(types))
            invoke = f'reinterpret_cast<{result} (*)({", ".join(types)})>(call.function)({arguments})'
            statement = invoke + ';' if result == 'void' else f'if ((*status = {invoke})) {{ *failed = i; return 0; }}'
            cases.append(f'case {index}: {statement} break;')
        source = '''#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
struct Call { void* function; void** arguments; size_t signature; size_t fill; };
struct Fill { void* data; size_t count; size_t width; unsigned char value[8]; };
template<size_t Width> void initialize(const Fill& fill) {
    auto* data = static_cast<unsigned char*>(fill.data);
    for (size_t i = 0; i < fill.count; ++i)
        std::memcpy(data + i * Width, fill.value, Width);
}
extern "C" double measure(Call* calls, size_t count, Fill* fills, size_t repeat, int32_t* status, size_t* failed) {
    if (repeat == 0) repeat = 1;
    auto begin = std::chrono::steady_clock::now();
    for (size_t rep = 0; rep < repeat; ++rep) {
        for (size_t i = 0; i < count; ++i) {
            const auto& call = calls[i];
            if (call.fill) {
                const auto& fill = fills[call.fill - 1];
                switch (fill.width) {
                case 1: initialize<1>(fill); break;
                case 2: initialize<2>(fill); break;
                case 4: initialize<4>(fill); break;
                case 8: initialize<8>(fill); break;
                }
                continue;
            }
            switch (call.signature) {
''' + '\n'.join(cases) + '''
            }
        }
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / static_cast<double>(repeat);
}
'''
        from ..output import worker_root
        root = worker_root() / 'native'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = path = Path(tempfile.mkdtemp(prefix='sequence-', dir=root))
        (path / 'timer.cpp').write_text(source)
        result = subprocess.run([*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17', '-O2', '-shared', '-fPIC',
                                 str(path / 'timer.cpp'), '-o', str(path / 'timer.so')], capture_output=True, text=True)
        if result.returncode:
            (path/'compiler.log').write_text(result.stdout+result.stderr)
            raise RuntimeError(f'native sequence timer compilation failed; see {path}: ' + result.stderr)
        self.library = ct.CDLL(str(path / 'timer.so'))
        self.measure = self.library.measure
        self.measure.argtypes = [ct.POINTER(_Call), ct.c_size_t, ct.POINTER(_Fill), ct.c_size_t, ct.POINTER(ct.c_int32), ct.POINTER(ct.c_size_t)]
        self.measure.restype = ct.c_double

    def run(self, repeat=1):
        repeat = max(1, int(repeat))
        for binding in self.bindings:
            binding.validate()
        status, failed = ct.c_int32(), ct.c_size_t()
        elapsed = self.measure(self.calls, len(self.calls), self.fills, repeat, ct.byref(status), ct.byref(failed))
        if status.value:
            self.bindings[failed.value].check_status(status.value, self.arguments[failed.value])
        for binding in self.bindings:
            binding.complete()
        return elapsed

    def close(self):
        self.measure = None
        self.library = None
        shutil.rmtree(self.directory)
