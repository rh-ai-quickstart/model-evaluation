{{/*
Expand the name of the chart.
*/}}
{{- define "model-evaluation.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "model-evaluation.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "model-evaluation.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "model-evaluation.labels" -}}
helm.sh/chart: {{ include "model-evaluation.chart" . }}
{{ include "model-evaluation.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "model-evaluation.selectorLabels" -}}
app.kubernetes.io/name: {{ include "model-evaluation.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "model-evaluation.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "model-evaluation.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image name helper
*/}}
{{- define "model-evaluation.image" -}}
{{- $registry := .Values.global.imageRegistry -}}
{{- $repository := .Values.global.imageRepository -}}
{{- $name := .name -}}
{{- $tag := .tag | default .Values.global.imageTag -}}
{{- printf "%s/%s/%s:%s" $registry $repository $name $tag -}}
{{- end }}

{{/*
API labels
*/}}
{{- define "model-evaluation.api.labels" -}}
{{ include "model-evaluation.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "model-evaluation.api.selectorLabels" -}}
{{ include "model-evaluation.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
UI labels
*/}}
{{- define "model-evaluation.ui.labels" -}}
{{ include "model-evaluation.labels" . }}
app.kubernetes.io/component: ui
{{- end }}

{{/*
UI selector labels
*/}}
{{- define "model-evaluation.ui.selectorLabels" -}}
{{ include "model-evaluation.selectorLabels" . }}
app.kubernetes.io/component: ui
{{- end }}

{{/*
Database labels
*/}}
{{- define "model-evaluation.database.labels" -}}
{{ include "model-evaluation.labels" . }}
app.kubernetes.io/component: database
{{- end }}

{{/*
Database selector labels
*/}}
{{- define "model-evaluation.database.selectorLabels" -}}
{{ include "model-evaluation.selectorLabels" . }}
app.kubernetes.io/component: database
{{- end }}

{{/*
Migration labels
*/}}
{{- define "model-evaluation.migration.labels" -}}
{{ include "model-evaluation.labels" . }}
app.kubernetes.io/component: migration
{{- end }}

{{/*
Migration selector labels
*/}}
{{- define "model-evaluation.migration.selectorLabels" -}}
{{ include "model-evaluation.selectorLabels" . }}
app.kubernetes.io/component: migration
{{- end }}

{{/*
Validate required values
*/}}
{{- define "model-evaluation.validateValues" -}}
{{- required "A valid .Values.secrets.API_TOKEN is required. Set it with --set secrets.API_TOKEN=<your-maas-token>" .Values.secrets.API_TOKEN -}}
{{- end }}

{{/*
Derive API base URL from route configuration
*/}}
{{- define "model-evaluation.apiBaseUrl" -}}
{{- if .Values.secrets.VITE_API_BASE_URL -}}
{{- .Values.secrets.VITE_API_BASE_URL -}}
{{- else if .Values.routes.enabled -}}
{{- $host := "" -}}
{{- if .Values.routes.sharedHost -}}
{{- $host = .Values.routes.sharedHost -}}
{{- else if .Values.routes.api.host -}}
{{- $host = .Values.routes.api.host -}}
{{- else if .Values.routes.ui.host -}}
{{- $host = .Values.routes.ui.host -}}
{{- end -}}
{{- if $host -}}
{{- if .Values.routes.api.tls.enabled -}}
{{- printf "https://%s/api" $host -}}
{{- else -}}
{{- printf "http://%s/api" $host -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end }}
