#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path


def run(command):
    subprocess.run(command, check=True)


def compiler_accepts(common: list[str], option: str, build_dir: Path) -> bool:
    probe_source = build_dir / "compiler_option_probe.cpp"
    probe_asm = build_dir / "compiler_option_probe.s"
    probe_source.write_text("int main() { return 0; }\n")
    return subprocess.run(
        [*common, "-S", option, str(probe_source), "-o", str(probe_asm)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def function_body(assembly: str, name: str) -> str:
    labels = [name, f"_{name}"]
    label_pattern = "|".join(re.escape(label) for label in labels)
    start = re.search(rf"^({label_pattern}):(?:\s|$).*$",
                      assembly, re.MULTILINE)
    if start is None:
        raise AssertionError(f"missing function label: {name}")
    label = start.group(1)
    tail = assembly[start.end():]
    end_positions = []
    for pattern in [
        rf"^\s*\.size\s+{re.escape(label)}\b",
        r"^\s*#+\s*-- End function\b",
        r"^LFE[0-9]+:",
    ]:
        end = re.search(pattern, tail, re.MULTILINE)
        if end is not None:
            end_positions.append(end.start())
    if not end_positions:
        raise AssertionError(f"missing function end marker: {name}")
    return tail[:min(end_positions)]


def require(pattern: str, text: str, message: str):
    if re.search(pattern, text, re.IGNORECASE) is None:
        raise AssertionError(message)


def reject(pattern: str, text: str, message: str):
    if re.search(pattern, text, re.IGNORECASE) is not None:
        raise AssertionError(message)


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, re.IGNORECASE))


def count_fma(text: str) -> int:
    return count(r"\bvfmadd\w*(?:sd|pd)\b", text)


def instruction_mnemonics(text: str) -> list[str]:
    mnemonics = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].split("//", 1)[0].strip()
        if not line or line.startswith(".") or line.endswith(":"):
            continue
        match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)\b", line)
        if match is not None:
            mnemonics.append(match.group(1).lower())
    return mnemonics


def check(args, build_dir: Path):
    asm_path = build_dir / "codegen_contract.s"
    exe_path = build_dir / "codegen_contract"
    common = [
        args.cxx,
        "-std=c++23",
        "-O3",
        "-march=native",
        "-DPOLYGNITION_TARGET_PROFILE=0",
        "-I",
        args.include,
    ]

    sources = [args.source]
    if args.bridge_target is not None:
        sources.append(args.bridge_target)

    run([*common, *sources, "-o", str(exe_path)])
    run([str(exe_path)])
    asm_command = [*common, "-S"]
    if compiler_accepts(common, "-masm=intel", build_dir):
        asm_command.append("-masm=intel")
    run([*asm_command, args.source, "-o", str(asm_path)])

    assembly = asm_path.read_text()
    static_motzkin = function_body(assembly, "polygnition_codegen_static_motzkin")
    algebra_static_motzkin = function_body(
        assembly, "polygnition_codegen_algebra_static_motzkin")
    runtime_motzkin = function_body(assembly, "polygnition_codegen_runtime_motzkin")
    automatic_degree12 = function_body(assembly, "polygnition_codegen_automatic_degree12")
    automatic_degree20 = function_body(assembly, "polygnition_codegen_automatic_degree20")
    prepared_degree20 = function_body(assembly, "polygnition_codegen_prepared_degree20")
    automatic_degree48 = function_body(assembly, "polygnition_codegen_automatic_degree48")
    complex_coeff_real_horner = function_body(
        assembly, "polygnition_codegen_complex_coeff_real_horner")
    static_bridge_degree20 = function_body(
        assembly, "polygnition_codegen_static_bridge_degree20")
    dorn2_degree20 = function_body(assembly, "polygnition_codegen_dorn2_degree20")
    dorn6_degree48 = function_body(assembly, "polygnition_codegen_dorn6_degree48")
    estrin_degree20 = function_body(assembly, "polygnition_codegen_estrin_degree20")
    batch_automatic_degree20 = function_body(
        assembly, "polygnition_codegen_batch_automatic_degree20")
    batch_prepared_degree20 = function_body(
        assembly, "polygnition_codegen_batch_prepared_degree20")
    batch_automatic_float_degree20 = function_body(
        assembly, "polygnition_codegen_batch_automatic_float_degree20")
    batch_preprocessed_motzkin = function_body(
        assembly, "polygnition_codegen_batch_preprocessed_motzkin")
    batch_knuth_split_degree12 = function_body(
        assembly, "polygnition_codegen_batch_knuth_split_degree12")

    reject(r"\b(v?divs[sd]|divs[sd]|call)\b", static_motzkin,
           "static Motzkin coefficients regressed to runtime preprocessing")
    reject(r"\b(v?divs[sd]|divs[sd]|call|j[a-z]+)\b", algebra_static_motzkin,
           "algebra-built static Motzkin regressed to runtime work")
    if instruction_mnemonics(algebra_static_motzkin) != instruction_mnemonics(static_motzkin):
        raise AssertionError(
            "algebra-built static quartic no longer matches direct literal instruction shape")
    require(r"\bv?divsd\b", runtime_motzkin,
            "runtime Motzkin preprocessing should still contain the coefficient division")
    if count(r"\bvmulsd\b", automatic_degree12) > 2:
        raise AssertionError(
            "automatic low-degree real evaluation should stay scalar Horner, not Dorn")
    reject(r"\b(call|j[a-z]+)\b", automatic_degree12,
           "automatic low-degree Horner evaluation should be straight-line code")
    if count(r"\bvmulsd\b", automatic_degree20) < 3:
        raise AssertionError(
            "automatic high-degree real evaluation did not materialize Dorn powers")
    if count_fma(automatic_degree20) < 8:
        raise AssertionError(
            "automatic high-degree real evaluation did not lower to Dorn FMA chains")
    reject(r"\b(call|j[a-z]+)\b", automatic_degree20,
           "automatic high-degree Dorn evaluation should be straight-line code")
    if (instruction_mnemonics(prepared_degree20) !=
            instruction_mnemonics(automatic_degree20)):
        raise AssertionError(
            "prepared point evaluation no longer matches direct evaluation")
    if count_fma(complex_coeff_real_horner) < 6:
        raise AssertionError(
            "complex-coefficient Horner at a real point should lower to scalar FMA chains")
    reject(r"\b(v?mulsd|vfnmadd\w*sd|call|j[a-z]+)\b",
           complex_coeff_real_horner,
           "complex-coefficient Horner should multiply by the real operand, not a lifted complex zero")
    reject(r"\bsub\s+rsp\b|\brep\s+movs\w*\b|\bmemcpy\b|\bv?mov\w*\s+(?:xmmword|ymmword|qword|dword)?\s*ptr\s*\[(?:rsp|rbp)",
           static_bridge_degree20,
           "passing a static literal by value should not materialize coefficient storage")
    if count_fma(automatic_degree48) < 20:
        raise AssertionError(
            "automatic higher-degree real evaluation did not lower to Dorn FMA chains")
    reject(r"\b(call|j[a-z]+)\b", automatic_degree48,
           "automatic higher-degree Dorn evaluation should be straight-line code")
    for body, name in [(dorn2_degree20, "dorn<2>"), (dorn6_degree48, "dorn<6>")]:
        if count_fma(body) < 8:
            raise AssertionError(f"explicit {name} evaluation did not lower to FMA chains")
        reject(r"\b(call|j[a-z]+)\b", body,
               f"explicit {name} evaluation should be straight-line code")
    if count_fma(estrin_degree20) < 6:
        raise AssertionError("explicit Estrin evaluation did not lower to FMA chains")
    reject(r"\b(call|j[a-z]+)\b", estrin_degree20,
           "explicit Estrin evaluation should be straight-line code")
    require(r"\bvfmadd\w*pd\b.*[xyz]mm", batch_automatic_degree20,
            "automatic batch evaluation should lower to packed FMA instructions")
    reject(r"\bcall\b", batch_automatic_degree20,
           "automatic batch evaluation should inline the vectorized loop body")
    if (instruction_mnemonics(batch_prepared_degree20) !=
            instruction_mnemonics(batch_automatic_degree20)):
        raise AssertionError(
            "prepared batch evaluation no longer matches direct evaluation")
    require(r"\bvfmadd\w*ps\b.*[xyz]mm", batch_automatic_float_degree20,
            "automatic float batch evaluation should lower to packed FMA instructions")
    reject(r"\bcall\b", batch_automatic_float_degree20,
           "automatic float batch evaluation should inline the vectorized loop body")
    require(r"\bvfmadd\w*pd\b.*[xyz]mm", batch_preprocessed_motzkin,
            "preprocessed Motzkin batch evaluation should lower to packed FMA instructions")
    reject(r"\bcall\b", batch_preprocessed_motzkin,
           "preprocessed Motzkin batch evaluation should inline the vectorized loop body")
    require(r"\bvfmadd\w*pd\b.*[xyz]mm", batch_knuth_split_degree12,
            "split real/imag Knuth batch evaluation should lower to packed FMA instructions")
    reject(r"\bcall\b", batch_knuth_split_degree12,
           "split real/imag Knuth batch evaluation should inline the vectorized loop body")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--bridge-target")
    parser.add_argument("--include", required=True)
    parser.add_argument("--build-dir", required=True)
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    check(args, build_dir)


if __name__ == "__main__":
    main()
