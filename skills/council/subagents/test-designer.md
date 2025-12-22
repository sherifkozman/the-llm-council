# Test Designer Subagent

## Purpose
Comprehensive test suite design with coverage goals and edge case identification.

## When to Use
- Designing test suites for new features
- Improving test coverage
- Creating test plans for complex logic
- Integration/E2E test strategies
- TDD planning

## Output Schema
Returns JSON with:
- `test_strategy`: Overall testing approach
- `test_scenarios`: Array of test cases with inputs/outputs
- `coverage_analysis`: What is/isn't covered
- `edge_cases`: Boundary conditions, error cases
- `test_implementation_guide`: Frameworks, fixtures, patterns
- `confidence`: Confidence score (0.0-1.0)

## CLI Options
```bash
council run test-designer "task" --health-check --json --verbose
```

## Example
```bash
council run test-designer "Design tests for cursor-based pagination" --json
```

## Cost & Time
- **Cost**: ~$0.29 per test plan
- **Time**: ~3.5 minutes
