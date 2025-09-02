# Branch Protection Guidelines for BensBot

To ensure code quality and stability, we recommend implementing the following branch protection rules in GitHub for the main branch(es).

## Recommended Branch Protection Settings

Navigate to `Settings > Branches > Branch protection rules` in your GitHub repository and add a new rule with these settings:

### For main/master/develop branches:

1. **Require pull request reviews before merging**
   - Require at least 1 approval
   - Dismiss stale pull request approvals when new commits are pushed
   - Require review from Code Owners

2. **Require status checks to pass before merging**
   - Require branches to be up to date before merging
   - Required status checks:
     - `lint-and-test`
     - `security-scan`
     - `build-test-containers`

3. **Require conversation resolution before merging**
   - All comments must be resolved before the PR can be merged

4. **Do not allow bypassing the above settings**
   - Ensure that administrators are also bound by these restrictions

5. **Restrict who can push to matching branches**
   - Only allow specific people or teams to push directly

6. **Allow force pushes**
   - Set to "Do not allow" to prevent history rewrites

7. **Allow deletions**
   - Set to "Do not allow" to prevent accidental branch deletion

## Additional Recommendations

1. **Protected Tags**
   - Protect release tags (e.g., `v*`) to prevent modification of releases

2. **Required Deployments**
   - Once you have deployment workflows set up, require successful deployments to staging environments before merging

3. **Require Linear History**
   - Consider enabling "Require linear history" to maintain a cleaner commit history

## CI/CD Integration

These branch protection rules work in conjunction with the CI/CD pipeline defined in `.github/workflows/ci.yml` to enforce:

- Code quality standards (linting, formatting)
- Passing tests
- Security scanning
- Container build validation

By enforcing these requirements, we ensure that every change to the codebase meets quality standards before being merged into protected branches.
