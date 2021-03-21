import React, { useState } from "react";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import TextField from "@material-ui/core/TextField";

import {
  InputHolder,
  StyledNewsletter,
  GettingStartedBox,
  ApiKey,
  InputError,
} from "@site/src/components/SignupForm.style";
import { Grid } from "@material-ui/core";

// Taken from https://ui.dev/validate-email-address-javascript/
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const trackEmailSignupSuccess = (isNewUser) => {
  ga("send", {
    hitType: "event",
    eventCategory: "API Register",
    eventAction: "Submit",
    eventLabel: isNewUser ? "New User" : "Existing User",
  });
};

const SignupForm = () => {
  const [email, setEmail] = useState();
  const [apiKey, setApiKey] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const { siteConfig } = useDocusaurusContext();

  const onSubmit = async (e) => {
    e.preventDefault();
    setApiKey("");
    if (!EMAIL_REGEX.test(email)) {
      setErrorMessage("Must supply a valid email address");
      return;
    }
    fetch(siteConfig.customFields.registerUrl, {
      method: "POST",
      body: JSON.stringify({ email }),
      headers: { "Content-Type": "application/json" },
    })
      .then((res) => res.json())
      .then((data) => {
        setErrorMessage(undefined);
        // Older API returned data json encoded in "body" parameter.
        setApiKey(data.api_key);
        trackEmailSignupSuccess(data.new_user);
      })
      .catch((err) => setErrorMessage("Must supply a valid email address"));
  };

  return (
    <GettingStartedBox>
      <p>
        There are just 3 questions we’d liked answered, and then you can
        immediately get started.{" "}
      </p>

      <div>
        <StyledNewsletter>
          <form>
            <Grid container spacing={4}>
              <Grid container item xs={12}>
                <Grid item xs={12}>
                  <h5>Email address</h5>
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    autoComplete="Email"
                    label="Email"
                    variant="outlined"
                    aria-label="Email"
                    required
                    fullWidth
                    id="fieldEmail"
                    type="email"
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </Grid>
              </Grid>
              <Grid container item xs={12}>
                <Grid item xs={12}>
                  <h5>Organization name</h5>
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    autoComplete="Organization"
                    aria-label="Organization"
                    fullWidth
                    variant="outlined"
                    label="Organization name"
                    id="fieldOrganization"
                    required
                  />
                </Grid>
              </Grid>

              <Grid container item xs={12}>
                <Grid item xs={12}>
                  <h5>How do you intend to our data?</h5>
                </Grid>
                <Grid item xs={12}>
                  <span>It’s optional, but it’s helpful for us to know:</span>
                  <ul>
                    <li>
                      The data/metrics you’re interested in (e.g. vaccine data,
                      risk levels, cases, deaths, etc.)
                    </li>
                    <li>
                      The locations you’d like to use this for (e.g. all 50
                      states, counties in the Florida panhandle, or just Cook
                      County, IL)
                    </li>
                    <li>How will this data be used to help people?</li>
                  </ul>
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    aria-label="How you are using the data"
                    placeholder="How are you using the data"
                    label="Use Case"
                    id="fieldUseCase"
                    type="email"
                    rows={5}
                    variant="outlined"
                    fullWidth
                    multiline
                  />
                </Grid>
              </Grid>
            </Grid>

            <button type="submit" onClick={(e) => onSubmit(e)}>
              Get API key
            </button>
            {errorMessage && <InputError>{errorMessage}</InputError>}
          </form>
        </StyledNewsletter>
        {!apiKey && (
          <p>
            If you've previously registered for an API key, you can use the form
            above to retrieve it.
          </p>
        )}
        {apiKey && (
          <p>
            Congrats, your new API key is <ApiKey>{apiKey}</ApiKey>
          </p>
        )}
      </div>
    </GettingStartedBox>
  );
};

export default SignupForm;
